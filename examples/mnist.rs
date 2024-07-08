use mml::eval::Evaluator;
use mml::optim::{AdamParams, Optimizer};
use mml::{eval, graph, optim, Tensor, W};

const PIXELS: usize = 28 * 28;
const CLASSES: usize = 10;
const SAMPLES: usize = 60_000;

fn main() {
    const BATCH: usize = 8;

    let data = mnist::MnistBuilder::new()
        .use_fashion_data()
        .label_format_one_hot()
        .download_and_extract()
        .finalize();

    assert_eq!(data.trn_img.len(), PIXELS * SAMPLES);
    assert_eq!(data.trn_lbl.len(), CLASSES * SAMPLES);

    let g = &mut graph::CGraph::new();
    let e = &mut eval::CPU::new();
    let o = &mut optim::Adam::new(g, AdamParams {
        lr: 0.001,
        ..Default::default()
    });

    let l1 = g.linear("l1", PIXELS, PIXELS);
    let l2 = g.linear("l2", PIXELS, PIXELS * 2);
    let l3 = g.linear("l3", PIXELS * 2, CLASSES);

    let inp = g.input_named("inp", [BATCH, 1, 1, PIXELS]);
    let exp = g.input_named("exp", [BATCH, 1, 1, CLASSES]);

    let v1 = l1(g, inp);
    let v1 = g.relu(v1);

    let v2 = l2(g, v1);
    let v2 = g.relu(v2);

    let v3 = l3(g, v2);

    let cls = g.softmax(W, v3);

    let loss = g.cross_entropy_loss(cls, exp);
    let loss = g.mean_reduce_all(loss);

    let max_loss = g.mean_reduce_all(loss);

    // We get list of parameters a result of backwards pass
    let params = g.backward(loss);

    for epoch in 0 .. 100000 {
        for bstart in (0..60_000 - BATCH).step_by(BATCH) {
            e.step();
            e.zero_grad(g, &params);

            // Set the inputs and expected output
            let ival = &data.trn_img[bstart * PIXELS..(bstart + BATCH) * PIXELS];
            let ival: Vec<f32> = ival.into_iter()
                .map(|v| *v as f32)
                .map(|v| v / 255.0 - 0.5)
                .collect();

            e.write(g, inp, &ival);

            let eval = &data.trn_lbl[bstart * CLASSES..(bstart + BATCH) * CLASSES];
            let eval: Vec<f32> = eval.into_iter().map(|v| *v as f32).collect();

            e.write(g, exp, &eval);

            e.eval(g, cls);
            e.eval(g, loss);

            o.optimize(g, e, &params);

            println!();
            println!("Epoch:\t{}", bstart);
            println!("Loss: \t{:?}", e.read(g, loss));
            println!("inp: \t{:?}", &e.read(g, inp)[0..10]);
            println!("Exp: \t{:?}", &e.read(g, exp)[0..10]);
            println!("v1: \t{:?}", &e.read(g, v1)[0..10]);
            println!("v2: \t{:?}", &e.read(g, v2)[0..10]);
            println!("l3.b: \t{:?}", &e.read(g, g["l3.bias"])[0..10]);
            println!("l3.bg:\t{:?}", &e.read(g, g[g["l3.bias"]].grad.unwrap())[0..10]);
            println!("l3.w: \t{:?}", &e.read(g, g["l3.weight"])[0..10]);
            println!("l3.wg:\t{:?}", &e.read(g, g[g["l3.weight"]].grad.unwrap())[0..10]);
            println!("v3: \t{:?}", &e.read(g, v3)[0..10]);
            println!("cls: \t{:?}", &e.read(g, cls)[0..10]);
            println!("cls(g):\t{:?}", &e.read(g, g[cls].grad.unwrap())[0..10]);

            e.eval(g, max_loss);
            let vmax = e.read(g, max_loss)[0];
            if vmax.abs() < 0.001 || vmax.is_nan() {
                println!("Seen:\t{:?} samples", (epoch * 60_000) + bstart * BATCH);
                return;
            }
        }
    }
}
