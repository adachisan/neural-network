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

var results = GetResults(50);
Task.WaitAll(results.ToArray());
results.Sort((a, b) => a.Result.Loss.CompareTo(b.Result.Loss));
for (var i = 0; i < 5; i++)
{
    var x = results[i].Result;
    Console.WriteLine($"{x.Epochs} {x.Loss}");
    x.Draw($"test[{i}]");
}


List<Task<Network>> GetResults(int len)
{
    return new Task<Network>[len].Select((task, i) =>
    {
        return Task.Run(() =>
        {
            var x = new Network(1, 10, 2, 1, Activation.ArcTan, 2);
            x.Fit(data);
            x.Learn();
            return x;
        });
    }).ToList();
}
