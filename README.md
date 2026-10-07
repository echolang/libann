# libann

You want a neural net in Echo. A small trainer, a runner, a file you can save and load. That is this library.

Tensors, layers, losses, optimizers, a trainer. A binary format for the whole model or for the weights alone. Pure Echo, no C. Layers, losses and optimizers are interfaces, so a new architecture is a type you write.

`ann::rl` sits on top: environments, replay and rollout buffers, and DQN, REINFORCE, actor-critic and PPO.

```bash
epm install
echoc test
echoc run -m . examples/xor.eco
```

The library has no program target. Those three commands are the whole checkout loop. `epm install` vendors [libcommand](https://github.com/echolang/libcommand), which draws the training output. `echoc build` without `-o` has nowhere to put a binary.

## A complete pass

Let's train XOR so the pieces have somewhere to sit.

```echo
use ann::{Act, BinaryCrossEntropy, Dataset, Dense, Log, Model, Sequential, Sgd, Tensor, Trainer};
use ann::cli;

array<array<float64>> $points = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
array<float64> $labels = [0.0, 1.0, 1.0, 0.0];
$data = Dataset(Tensor(rows: $points), Tensor(column: $labels));

$net = Sequential()
    ->add(Dense(2, 8))
    ->add(Act(.tanh))
    ->add(Dense(8, 1))
    ->add(Act(.sigmoid));

$model = Model($net, $seed: 7);
```

`Model($net, $seed: 7)` wraps the chain and fills every param from that seed. Building a `Dense` leaves the weights at zero until then. Same architecture and seed always start from the same weights.

The network ends in `Act(.sigmoid)` because this is a yes/no. `BinaryCrossEntropy` is the loss that goes with that.

```echo
$log = Log()->to(cli::Console());
$log->begin('xor');
$log->model($model);

$trainer = Trainer(
    $epochs: 2000,
    $batchSize: 4,
    $metric: .binaryAccuracy,
    $observer: $log->training(),
);
$trainer->fit($model, BinaryCrossEntropy(), Sgd(0.5, $momentum: 0.9), $data);
```

Then you have a network. Save the architecture and the weights, load them back, run them.

```echo
guard $model->save('xor.ann') else ($e) {
    die("save failed: {$e}");
}

Model $loaded = guard Model::load('xor.ann') else ($e) {
    die("load failed: {$e}");
}

$out = $loaded->predict($data->inputs);
$classes = $loaded->classify($data->inputs);
```

`classify` reads a single output as class 1 at 0.5 and above. Several outputs: largest wins.

`save` / `load` are architecture plus weights. Built-in kinds are already on `Registry()`. A kind you wrote needs one `add` first; that is later.

The whole file is `examples/xor.eco`.

## From another module

```bash
epm add --path ../libann
```

That writes a `#[depends:]` line. Echo sees `namespace ann` as soon as the module loads. There is nothing to link. libann's own `#[requires:]` (libcommand) comes along: run `epm install` in your program and it lands in your `vendor/`.

```echo
use ann::{Act, Dense, Model, Sequential};
```

Examples from this tree stay `echoc run -m . examples/<file>.eco`. A release binary of an example needs an output path, because the manifest is a library:

```bash
echoc build -m . -o xor examples/xor.eco
```

`echoc build` is release: `assert` and the runtime bounds checks vanish. Shape bugs and programmer mistakes in this library `die`. That is on purpose. A check that only lives in debug would ship a silent wrong network.

## Networks

`Sequential` is a `Layer`. Chains nest, residuals wrap chains, a custom layer sits in the same list.

```echo
$net = Sequential()
    ->add(Dense(2, 8))
    ->add(Act(.tanh))
    ->add(Dense(8, 1))
    ->add(Act(.sigmoid));
```

You can also pass the list at construction: `Sequential($layers: [$dense, $act])`.

### Dense

Fully connected: `output = input · weight + bias`.

```echo
Dense(2, 8)           // xavier, the default
Dense(4, 64, .he)     // before ReLU
```

Weights stay zero until `Model` calls `initialize`. Bias always starts at zero. Pick the init by the activation that follows: `.xavier` for tanh and sigmoid, `.he` for the ReLU family, `.lecun` for GELU.

### Act

An `Activation` on every value. No params.

```echo
Act(.relu)
Act(.gelu)
```

The set: `.linear`, `.relu`, `.leakyRelu`, `.sigmoid`, `.tanh`, `.gelu`, `.softplus`, `.elu`, `.swish`. A new activation is a new case and two arms.

### Dropout

Inverted dropout. Training zeroes each value with probability `rate` and scales the survivors by `1 / (1 - rate)`, so inference is the identity.

```echo
Dropout(0.1)
```

`rate` must be in `[0, 1)`.

### LayerNorm

Per sample: zero mean, unit variance over features, then learned `gamma` and `beta`. Same in train and infer, unlike batch norm. The normalisation of transformer blocks.

```echo
LayerNorm(32)
```

### Residual

Skip connection: `output = input + inner(input)`. The inner layer must keep the shape.

```echo
$block = Sequential()
    ->add(Dense(32, 32, .he))
    ->add(LayerNorm(32))
    ->add(Act(.gelu));

Residual($block)
```

### Softmax

Row-wise: each sample's scores become probabilities. Put it at the end of a classifier that should output probabilities.

Here is the catch. Training with `CrossEntropy`: leave it out. That loss applies softmax itself, so the value and the gradient stay numerically stable, and `CrossEntropy::predict` gives the probabilities back. `Model::predict` on a `CrossEntropy` network returns logits.

## Data and training

A `Tensor` is a shaped block of float64, row-major. Batches go first: 32 samples of 4 features is `[32, 4]`. Layers treat everything after the first axis as one row, so `[batch, n]` code still works on extra axes.

```echo
Tensor(rows: [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
Tensor(column: [0.0, 1.0, 1.0, 0.0])
Tensor(oneHot: $labels, classes: 3)
```

`Tensor(rows: ...)` is the readable way to write a dataset by hand. Every row must be as long as the first. A count that does not fill the shape is a bug in the caller and stops the program.

### Dataset

Samples, targets and per-sample weights, kept side by side so shuffle, batch and split never pull them apart.

```echo
$data = Dataset(Tensor(rows: $points), Tensor(column: $labels));
$split = $data->split(0.2, $seed: 7);
```

Default weight is 1, the plain batch mean. Other weights are how RL talks to a loss: an advantage per sample, an importance weight for prioritised replay. Negative weights push a sample's loss up instead of down.

`split` shuffles, then `$fraction` of the samples go to `holdout`, the rest to `train`.

A `Standardizer` is zero mean, unit deviation per column. Fit on training data, then apply unchanged to everything else:

```echo
$scale = Standardizer(fit: $split->train->inputs);
```

### Trainer

Mini-batch gradient descent over a `Dataset`. Every batch is one `Model::learn`. A loop of your own can call that, or skip it and drive `forward` / `backward` / `step` by hand.

```echo
$trainer = Trainer(
    $epochs: 300,
    $batchSize: 16,
    $metric: .rmse,
    $validation: $split->holdout,
);
$history = $trainer->fit($model, MeanSquaredError(), Adam(0.01), $split->train);
```

`fit` returns every epoch's `EpochReport` in a `History`. `$onEpoch` is a closure called after each one, and `$observer` hears every batch as well. Both are optional.

Metrics are reported next to the loss. Training never optimises them. `.accuracy` is the share of samples whose largest output is the target class (one-hot rows or a `[n, 1]` column of class indices). `.binaryAccuracy` is the share of values on the same side of 0.5 as their target. `.rmse` and `.mae` are in the units of the target.

`rate` is public on the built-in optimizers. A schedule is one assignment in `onEpoch`:

```echo
if ($r->epoch % 50 == 0) {
    $adam->rate = $adam->rate * 0.5;
}
```

`$clip` caps the global L2 of the gradients. 0 leaves them alone. The last batch of an epoch may be smaller than `$batchSize`.

### Losses

What training minimises. Scores each sample on its own; averaging and weighting live in `Dataset`. That keeps a custom loss short, and every loss gets per-sample weights for free.

Regression: `MeanSquaredError`, `MeanAbsoluteError`, or `Huber`. Binary classifier ending in sigmoid: `BinaryCrossEntropy`. Multi-class ending in a plain `Dense` (logits): `CrossEntropy`. `smoothing` mixes the targets with uniform. Cheap regulariser for an overconfident classifier.

### Optimizers

`Sgd` with optional Nesterov momentum. `Adam` is the usual default; non-zero `weightDecay` makes it AdamW.

```echo
Sgd(0.5, $momentum: 0.9)
Adam(0.01, $weightDecay: 0.0001)
```

### When Trainer is the wrong loop

A policy gradient, a TD error, anything whose objective is not "match these labels": `learn` is still one supervised step, and you can skip it.

```echo
Tensor $out = $model->forward($inputs, .train);
$model->backward($gradientOfTheObjective);
$model->step($optimizer, $clip: 0.5);
```

`.train` caches what `backward` needs. `.infer` must leave those caches alone, so a critic or Double DQN can peek between a training forward and its backward. Several `backward`s before one `step` add up. `step` clips (when `$clip` is above zero), lets the optimizer move every param, then clears the gradients.

## Save and load

Two file kinds.

A model (`save` / `load`) is architecture plus weights, rebuilt from a `Registry`. Magic `ANNM`.

```echo
guard $model->save('xor.ann') else ($e) {
    die("save failed: {$e}");
}

Model $loaded = guard Model::load('xor.ann') else ($e) {
    die("load failed: {$e}");
}
```

Weights (`saveWeights` / `loadWeights`) load into a network you built in code. Magic `ANNW`. Every name and shape must match or nothing changes.

```echo
$fresh = Model(network(), $seed: 1234);
guard $fresh->loadWeights('spiral.weights') else ($e) {
    die("loadWeights failed: {$e}");
}
```

That is how you ship an architecture in source and only the numbers in a file. `examples/spiral.eco` does exactly this.

`copy` overwrites every param from another model of the same architecture. `blend` does it as a fraction, the way DQN trails its target with `tau`. `clone` round-trips through the codec, so you hold a second network.

Failures come back as `result<Model, ModelError>` / `status<ModelError>`. Wrong magic, truncated bytes, unknown layer kind, a param whose name or shape does not match: you get the case, not a half-applied network.

## A layer the library has never heard of

The extension point of the whole library is `Layer`. Implement `forward`, `backward` and `params`, and the rest (train, save, load, nest) comes for free.

`kind` is the registry name, written in front of the layer's config. `encode` writes config only, not weights. The decoder for that kind rebuilds an equal layer. `initialize` is called once by `Model` with a seeded generator. Construction itself does no random work.

`Registry()` already knows the built-ins. A custom layer is one `add` away from loading like them:

```echo
$registry = Registry()->add('gate', function(Decoder $in, Registry $r) : result<Layer, ModelError> {
    return Gate::decode($in, $r);
});

Model $loaded = guard Model::load('gate.ann', $registry) else ($e) {
    die("load failed: {$e}");
}
```

The decoder gets the registry so a composite can decode children with `$r->decode($in)`, whatever kinds they are.

`examples/custom_layer.eco` is a learned per-feature gate: trained, saved, loaded back.

## Reinforcement learning

The gradient does not come from a labelled dataset. You still use `Model`. You supply the objective.

Everything in `ann::rl` talks through `Environment`: observations are `[1, features]`, actions are a fixed discrete set, and each step answers with a reward and whether the episode goes on (`running`, `terminated`, `truncated`). After `terminated` there is no future to bootstrap from. After `truncated` there is; the clock just ran out.

`Runner` drives that one step at a time: current observation, automatic resets, return of every finished episode. `$observation` is a public field. `step` writes it; everyone else reads it.

CartPole is bundled. Observation `[1, 4]`, action 0 left / 1 right, fail past 12 degrees or 2.4 m from centre, reward 1 per step survived, cut-off at 500. Mean return 195 is the traditional "solved"; with the default limit the best possible is 500.

### DQN

A network estimates the value of every action, and learns from replay to match reward plus discounted value of the best next action. `online` acts and learns. `target` only supplies next-state values.

```echo
use ann::rl::{CartPole, Dqn, Replay, Report, Runner, Schedule};

$runner = Runner(CartPole($seed: 1));
$dqn = Dqn(Model(qnet(), $seed: 1), Model(qnet()), Adam(0.001), $tau: 0.005);
$replay = Replay(50000);
$epsilon = Schedule(from: 1.0, to: 0.02, over: 10000);
$rng = Random(1);

for (usize $i = 0; $i < 60000; $i++) {
    $replay->add($runner->step($dqn->act($runner->observation, $epsilon->at($i), $rng)));

    if ($replay->count() >= 1000) {
        Report $r = $dqn->update($replay->sample(64, $rng));
    }
}
```

With `double` on (the default), online picks the next action and target values it. `tau` above zero blends a fraction of online into target after every update. Otherwise the target copies every `sync` updates.

### Policy gradients

`Policy` is an ordinary `Model` ending in a plain `Dense`: observations in, one logit per action, read as a softmax. The critic of actor-critic and PPO is a separate `Model` with one output.

`Reinforce` measures "better" by the discounted return, policy only. `ActorCritic` measures it against a critic with GAE. Both learn from a `Rollout` of the current policy and expect it cleared afterwards.

`Ppo` squeezes several epochs of minibatch updates out of every rollout, and clips the probability ratio so the policy cannot drift far from the one that collected it. Record each action's log-probability in the rollout (`Rollout::add($t, $choice->logProb)`); PPO needs it.

```echo
Choice $c = $policy->act($runner->observation, $rng);
$rollout->add($runner->step($c->action), $c->logProb);
Report $r = $agent->update($rollout);
$rollout->clear();
```

`examples/cartpole_dqn.eco`, `examples/cartpole_a2c.eco`, `examples/cartpole_ppo.eco` are the three loops in full.

A game, a simulator or a business process plugs in by implementing four methods on `Environment`. Randomness belongs to the environment: seed it on construction so runs reproduce.

## On the GPU

A wide MLP outgrows the CPU fast: three 1024-wide layers at batch 256 take 160 ms a step here. `metal/` is a separate module, `libann-metal`, that mirrors a `Model` onto the GPU through Metal, trains it there in float32, and syncs the weights back into the same model. Same trainer fields, same `History`, same files. Apple platforms only, Dense and Act layers only for now, and under about 128 wide the CPU is still faster.

It's a module of its own so this one stays pure Echo with nothing to link. [metal/README.md](metal/README.md) has the numbers and the rest.

## Catalog

### Layers

#### `Dense`

Fully connected. `Dense($inputs, $outputs, $init = .xavier)`.

#### `Act`

Elementwise `Activation`. No params.

#### `Dropout`

Inverted dropout. `rate` in `[0, 1)`.

#### `LayerNorm`

Per-sample normalisation, learned scale and shift.

#### `Residual`

`input + inner(input)`. Inner keeps the shape.

#### `Softmax`

Row-wise probabilities. Skip it when training with `CrossEntropy`.

#### `Sequential`

A chain of layers, itself a `Layer`.

#### `Conv2D`

2D convolution over HWC images stored flat, one image per row. `Conv2D($channels, $filters, $height, $width)`, 3x3 stride 1 padding 1 by default. Im2col plus one matmul.

#### `MaxPool2D`

Max over non-overlapping windows. `MaxPool2D($channels, $height, $width)`, window 2.

#### `SelfAttention`

Multi-head self-attention over the slots of a row. `SelfAttention($slots, $width, $heads: n)`. No residual or norm inside.

#### `Parallel`

Branches over column ranges of the same input, outputs side by side. `pass($from, $count)` copies a range unchanged.

#### `PerSlot`

One shared layer over equal-width slots packed in a row. The phi of Deep Sets.

#### `Pool`

Collapses slots into a mean, a max, or both. `Pool($slots, .meanMax)`. No params.

#### `Broadcast`

Hands a shared context to every slot. After a pooled summary sitting in front of the slots.

#### `Permute`

Reorders columns. `Permute(slots: n, width: w)` regroups slot-major into field-major.

### Activations

`.linear`, `.relu`, `.leakyRelu` (0.01x on the negative side, so no unit is stuck at zero gradient), `.sigmoid`, `.tanh`, `.gelu`, `.softplus`, `.elu`, `.swish`.

### Losses

#### `MeanSquaredError`

Default for regression.

#### `MeanAbsoluteError`

Less swayed by outliers than squared error.

#### `Huber`

Squared up to `delta`, absolute beyond. The usual TD loss of DQN. Default `delta` is 1.

#### `BinaryCrossEntropy`

Probabilities in (0, 1). Pair with a final `Act(.sigmoid)`.

#### `CrossEntropy`

Softmax cross-entropy over logits. Targets are one-hot rows, or soft distributions. `smoothing` in `[0, 1)`.

#### `SigmoidCrossEntropy`

Independent logistic losses over logits. Pair with a plain `Dense` when yes/no heads share a linear layer.

#### `Heads`

Several losses over slices of one output. `add($width, $loss)` in column order; `gate:` scores only rows whose extra target column holds 1.

### Optimizers

#### `Sgd`

`$rate = 0.01`, `$momentum = 0.0`, `$nesterov = false`.

#### `Adam`

`$rate = 0.001`. Non-zero `$weightDecay` is AdamW.

### Metrics

`.none`, `.accuracy`, `.binaryAccuracy`, `.mae`, `.rmse`.

### Inits

`.zeros` (right for biases, wrong for weights: every unit would learn the same thing), `.ones`, `.uniform` in `[-0.05, 0.05]`, `.xavier`, `.he`, `.lecun`.

### Examples

```bash
echoc run -m . examples/xor.eco
echoc run -m . examples/regression.eco
echoc run -m . examples/spiral.eco
echoc run -m . examples/custom_layer.eco
echoc run -m . examples/cartpole_dqn.eco
echoc run -m . examples/cartpole_a2c.eco
echoc run -m . examples/cartpole_ppo.eco
echoc run -m . examples/multihead.eco
echoc run -m . examples/sets.eco
echoc run -m . examples/shapes.eco
```

`examples/bench.eco` is the hot paths. Release builds only; the JIT's debug checks would measure themselves:

```bash
echoc build -m . --optimize whole -o /tmp/annbench examples/bench.eco && /tmp/annbench
```

## License

MIT.
