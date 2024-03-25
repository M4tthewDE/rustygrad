use rustygrad::models::efficientnet::Efficientnet;

fn main() {
    let classes = 10;
    let num = 0;
    let model = Efficientnet::new(num, classes, false);
}
