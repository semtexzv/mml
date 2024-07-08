use mml::eval::Evaluator;
use mml::optim::Optimizer;
use mml::{eval, graph, optim, prod, Tensor};

/// Models a network expressing a * x + b
fn model(g: &mut graph::CGraph, batch: usize) -> [Tensor; 4] {
    // Params need to be named
    let a = g.param("a", [1, 1, 1, 1]);
    let b = g.param("b", [1, 1, 1, 1]);

    // Inputs need to be named
    let x = g.input_named("x", [batch, 1, 1, 1]);
    let z = g.input_named("z", [batch, 1, 1, 1]);

    // No operators, everything is a method on CGraph
    let y = g.mul(a, x);
    let y = g.add(y, b);

    let y = g.named("y", y);

    // MSE Loss = (y - z) ** 2
    let loss = g.sub(y, z);
    let loss = g.mul(loss, loss);
    let loss = g.sum_reduce_all(loss);
    let batch = g.val(prod(g[y].sh) as f32, [1, 1, 1, 1]);

    let loss = g.div(loss,  batch);

    // [input, output, loss] tensors
    [x, z, y, loss]
}

fn target(x: f32) -> f32 {
    32.0 * x + 10.0
}

fn main() {
    const BATCH: usize = 16;

    let g = &mut graph::CGraph::new();
    let e = &mut eval::CPU::new();
    let o = &mut optim::Adam::new(g, Default::default());

    let [t_inputs, t_expect, t_output, t_loss] = model(g, BATCH);

    let a = g.find("a");
    let b = g.find("b");

    // We get list of parameters a result of backwards pass
    let params = g.backward(t_loss);

    // Auxiliary tensor for logging, max loss in batch
    let max_loss = g.max_reduce(mml::B, t_loss);

    for epoch in 0.. {
        // Evaluator is lazy, we need to step before every iteration
        e.step();
        e.zero_grad(g, &params);

        let input: [f32; BATCH] = rand::random();
        let expect: [f32; BATCH] = input.map(|s| target(s));

        // We write into input and output tensors, marking them dirty.
        e.write(g, t_inputs, &input);
        e.write(g, t_expect, &expect);

        // Calculate loss, will re-evaluate dirty tensors.
        e.eval(g, t_loss);
        e.eval(g, max_loss);
        if e.read(g, max_loss)[0] < 1e-6 {
            println!("Seen:\t{} samples, converged to {:?}", epoch * BATCH, e.read(g, t_loss));
            break;
        }
        o.optimize(g, e, &params);

        println!("Epoch\t{:?}", epoch);
        println!("Inpu \t{:?}", e.read(g, g.find("x")));
        println!("Expc \t{:?}", e.read(g, g.find("z")));
        println!("Loss \t{:?}", e.read(g, t_loss));
        println!("Prm \t{:?} {:?}", e.read(g, a), e.read(g, b));
        println!("GRD \t{:?} {:?}", e.read(g, g[a].grad.unwrap()), e.read(g, g[b].grad.unwrap()));
    }
}
