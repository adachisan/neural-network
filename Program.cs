// https://playground.tensorflow.org/
// https://csv2chart.com/

using NeuralNetwork;
using static NeuralNetwork.Network.Neuron;

var data = new (float[], float[])[]
{
    (new [] { 0f }, new [] { 0f }),
    (new [] { 0.125f }, new [] { 1f }),
    (new [] { 0.25f }, new [] { 0f }),
    (new [] { 0.375f }, new [] { 1f }),
    (new [] { 0.5f }, new [] { 0f }),
    (new [] { 0.625f }, new [] { 1f }),
    (new [] { 0.75f }, new [] { 0f }),
    (new [] { 0.875f }, new [] { 1f }),
    (new [] { 1f }, new [] { 0f })
};

var avg = 0f;
var count = 0f;
Parallel.For(0, 99, (i, s) =>
{
    var x = new Network(1, 10, 2, 1, ActivationType.Tanh, ActivationType.None, 2);
    x.Fit(data);

    if (!s.IsStopped)
    {
        x.Learn();
        avg += x.Loss;
        count++;
        if (x.Loss < 0.1f)
        {
            Console.WriteLine($"=> [{i}] {x.Epochs} {x.Loss:n2}");
            x.Save("data");
            s.Stop();
        }
        else Console.WriteLine($"[{i}] {x.Epochs} {x.Loss:n2}");
    }
});

Console.WriteLine($"Average_loss: {avg / count:n2}");

var x = new Network("data");
x.Draw("data");
