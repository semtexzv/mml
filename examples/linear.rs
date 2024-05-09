use rand::random;
use mml::{graph, eval, optim, Tensor};
use mml::eval::Evaluator;
use mml::optim::Optimizer;

/// Models a network expressing a * x + b
fn model(g: &mut graph::CGraph, batch: usize) -> [Tensor; 3] {

    // Params need to be named
    let a = g.param("a", [1, 1, 1, 1]);
    let b = g.param("b", [1, 1, 1, 1]);

    // Inputs need to be named
    let x = g.input("x", [batch, 1, 1, 1]);
    let z = g.input("z", [batch, 1, 1, 1]);

    // No operators, everything is a method on CGraph
    let y = g.mul(a, x);
    let y = g.add(y, b);

    // MSE Loss = (y - z) ** 2
    let loss = g.sub(y, z);
    let loss = g.mul(loss, loss);

    // [input, output, loss] tensors
    [x, z, loss]
}


fn target(x: f32) -> f32 {
    32.0 * x + 10.0
}

fn main() {
    const BATCH: usize = 32;

    let mut g = &mut graph::CGraph::new();
    let mut e = &mut eval::CPU::new();
    let mut o = &mut optim::SGD::new(g, 0.01);

    let [inp, out, loss] = model(g, BATCH);

    let a = g.find("a");
    let b = g.find("b");

    // We get list of parameters a result of backwards pass
    let params = g.backward(loss);

    // Auxiliary tensor for logging, max loss in batch
    let max_loss = g.max_reduce(mml::B, loss);

    for epoch in 0.. {
        // Evaluator is lazy, we need to step before every iteration
        e.step();

        let sample: [f32; BATCH] = random();

        // We write into input and output tensors, marking them dirty.
        e.set_value(g, inp, &sample);
        e.set_value(g, out, &sample.map(|s| target(s)));

        // Calculate loss, will re-evaluate dirty tensors.
        e.evaluate(g, loss);
        println!("Epoch\t{:?}", epoch);
        println!("Loss:\t{:?}", e.get_value(loss)[0]);
        println!("Params:\t{:?} {:?}", e.get_value(a), e.get_value(b));

        e.evaluate(g, max_loss);
        if e.get_value(max_loss)[0] < 0.0000001 {
            println!("Seen:\t{} samples", epoch * BATCH);
            println!("Params:\t{:?} {:?}", e.get_value(a), e.get_value(b));
            break;
        }
        o.optimize(g, e, &params);
    }
}
