# libann-metal

You have a libann model that got big. 512 wide, 1024 wide, batches in the hundreds, and every epoch takes coffee. This module trains it on the GPU through Metal, then hands the weights back to the same `ann::Model` you started with.

Nothing about the model changes. You build it with libann, mirror it onto the device, train there, and `save` / `load` / `predict` on the CPU like always. The core library stays pure Echo; this is the opt-in part that links Metal.

Apple platforms only. Everywhere else `ann::metal::available()` is false and nothing else in `ann::metal` exists.

```bash
cd metal
epm install
echoc test --package-dir vendor
cd examples
echoc run --package-dir ../vendor --target parity
echoc run --package-dir ../vendor --target classify
```

`epm install` vendors [libmetal](https://github.com/echolang/libmetal) and libann's own libcommand. The examples are their own module, so they point at `../vendor`.

## Is it worth it

Depends entirely on the width. A GPU step costs a fixed few hundred microseconds before it does any math, and a small network is done on the CPU by then. Three dense layers, forward + backward + Adam, M2 Max, `examples/bench.eco`:

| width | batch | cpu ms | gpu ms | speedup |
|---|---|---|---|---|
| 64 | 64 | 0.19 | 0.39 | 0.5x |
| 128 | 64 | 0.67 | 0.47 | 1.4x |
| 256 | 64 | 2.48 | 0.59 | 4.2x |
| 512 | 64 | 10.2 | 0.87 | 12x |
| 1024 | 64 | 43.6 | 1.43 | 30x |
| 2048 | 64 | 332 | 3.32 | 100x |
| 256 | 256 | 10.1 | 0.82 | 12x |
| 1024 | 256 | 161 | 2.02 | 80x |
| 2048 | 256 | 1337 | 5.26 | 254x |

So: under about 128 wide, stay on the CPU. The CartPole examples in libann (64 wide, acting one observation at a time) are exactly the case where the GPU loses. From 256 up it isn't close.

```bash
cd metal/examples
echoc build --package-dir ../vendor --target bench --optimize whole -o /tmp/annmetalbench && /tmp/annmetalbench
```

## A complete pass

```echo
use ann::{Act, Adam, CrossEntropy, Dense, Model, Sequential};
use ann::metal::{Gpu, Objective, Trainer, Update};

$gpu = guard Gpu::open() else ($e) {
    die("{$e}");
}

$model = Model(Sequential()
    ->add(Dense(64, 512, .he))
    ->add(Act(.gelu))
    ->add(Dense(512, 512, .he))
    ->add(Act(.gelu))
    ->add(Dense(512, 10)), $seed: 1);

$net = guard $gpu->mirror($model, $batch: 256) else ($e) {
    die("{$e}");
}

$history = guard Trainer($epochs: 10, $batchSize: 256, $metric: .accuracy)
    ->fit($net, Objective::crossEntropy(CrossEntropy()), Update::adam(Adam()), $data) else ($e) {
    die("{$e}");
}

guard $model->save('classifier.ann') else ($e) {
    die("save failed: {$e}");
}
```

`Gpu::open` compiles the kernels. Do it once per program; every network shares them.

`mirror` copies the model onto the device and sizes everything for batches of up to `$batch` rows. It's also where an unsupported layer is caught, before any training happens.

`ann::metal::Trainer` has the same fields as `ann::Trainer`: epochs, batch size, shuffle, seed, metric, clip, validation, `onEpoch`, `observer`. Defaults match too, including `batchSize` 32. It returns the same `History`, so `Log::training()` draws it the same way. The difference: the dataset is uploaded once, and every batch is gathered on the device. An epoch never copies a sample.

After every epoch, `fit` writes the weights into the host model before `onEpoch` runs, so a checkpoint there just works. When `fit` returns, `$model` already holds the trained weights. That's why `save` just works.

### Objective and Update

Here's the catch. The GPU can't run an arbitrary `Loss` or `Optimizer`, those are interfaces with Echo code inside. So you hand it the built-in object wrapped in what it knows how to run:

```echo
Objective::mse(MeanSquaredError())
Objective::huber(Huber($delta: 0.5))
Objective::crossEntropy(CrossEntropy($smoothing: 0.1))

$adam = Adam(0.001, $weightDecay: 0.01);
Update::adam($adam)
Update::sgd(Sgd(0.05, $momentum: 0.9, $nesterov: true))
```

`Update` keeps the optimizer object and reads its settings on every step. A schedule that sets `$adam->rate` in `onEpoch` works exactly like on the CPU. The running state (Adam's moments, SGD's velocity) lives on the device, with the network.

`Objective` keeps the loss for the CPU-side work: `predict` for the metric, and scoring a validation set.

## Your own loop

`Trainer` is the common case. `Network::learn` is one step, same shape as `Model::learn`:

```echo
$loss = $net->learn($batch, Objective::mse(MeanSquaredError()), Update::adam($adam), $clip: 1.0);
$out = $net->predict($inputs);
$net->sync();
```

`learn` uploads the batch, runs forward, loss, backward, clipping and the optimizer as one command buffer, waits once, and returns the loss from before the step.

`predict` takes any number of rows and walks them `capacity` at a time. It returns the raw output; pass it through `objective->loss()->predict` for probabilities from a cross-entropy net.

The weights live on the device while you train. The host model doesn't see them until you `sync`. Changed the host model yourself, with a `loadWeights` or a `copy`? `reload` pushes it back up.

## What runs on the GPU

`Dense` and `Act`, inside any nesting of `Sequential`. All nine activations. That's an MLP, which is what gets wide enough for this to matter.

`Dropout`, `LayerNorm`, `Residual`, `Softmax` and custom layers don't run yet. `mirror` fails with `GpuError::unsupported($index)`, the index counting through the network with every `Sequential` opened. A net with no Dense at all (empty, or Act only) is `GpuError::noDense`. A `Dense` that doesn't take the width the layer before it gives is `GpuError::shape($index)`.

Losses: all five built-ins. Optimizers: `Sgd` (plain, momentum, Nesterov) and `Adam` (and AdamW). Global gradient clipping too.

## Numbers

The GPU computes in float32, the CPU in float64. The host model stays float64; `sync` converts on the way back, so the file format doesn't change.

`examples/parity.eco` trains the same network from the same seed on both, for every loss and optimizer, and stops on the first disagreement. Over 20 steps the losses stay within about 2e-6 of each other, the weights within about 5e-7. Feed it inputs around ±30 and the weights drift further apart, around 5e-4. That's Adam: where a gradient is nearly zero, float32 and float64 can disagree on its sign, and Adam turns either into a whole step.

`BinaryCrossEntropy` clips probabilities at 1e-7 rather than the CPU's 1e-12. 1 - 1e-12 is just 1 in float32.

## Testing

`echoc test` forks, and the Metal shader compiler (XPC) doesn't survive a fork. So the tests check what doesn't need it (layer validation, the loss codes, buffer round trips through float32) and skip the GPU step with a note. `examples/parity` is the real GPU test. Run it after touching a kernel.

## From another module

```bash
epm add --path ../libann/metal
epm install
```

The path module brings libann with it (`#[depends: ".."]`), and `epm install` at your program vendors libmetal into your `vendor/`.

On a program that also builds for other platforms, gate the code that names `ann::metal` types on the OS, and check `ann::metal::available()` at runtime for a Mac without a Metal device.

## License

MIT.
