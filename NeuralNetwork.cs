namespace NeuralNetwork
{
    using System.Text;
    using System.Text.Json;
    using static Network.Neuron;

    // https://playground.tensorflow.org/
    public class Network
    {
        public Neuron[][] Hidden { get; set; }
        public Neuron[] Output { get; set; }
        public float Loss { get; set; }
        public string Epochs { get; set; }
        List<(float[] inputs, float[] target)> Data = new List<(float[] inputs, float[] target)>();

        public Network() { }

        public Network(int inputs, int neurons, int layers, int outputs, ActivationType hidden, ActivationType output, float range = 2)
        {
            Hidden = Enumerable.Repeat(neurons, layers).Select(n => new Neuron[n]).ToArray();
            for (int l = 0; l < layers; l++)
                for (int n = 0; n < neurons; n++)
                    Hidden[l][n] = new Neuron((l == 0) ? inputs : neurons, hidden, range);
            Output = Enumerable.Range(0, outputs).Select(o => new Neuron(neurons, output, range)).ToArray();
        }

        public Network(string file)
        {
            var json = File.ReadAllText($"{file}.json");
            var data = JsonSerializer.Deserialize<Network>(json);
            Hidden = data.Hidden;
            Output = data.Output;
            Loss = data.Loss;
        }

        public void Fit(params (float[] inputs, float[] target)[] data)
        {
            foreach (var item in data)
                Data.Add(item);
        }

        public float[] Predict(params float[] inputs)
        {
            foreach (var layer in Hidden)
            {
                Array.ForEach(layer, n => n.Update(inputs));
                inputs = layer.Select(n => n.Output).ToArray();
            }
            Array.ForEach(Output, o => o.Update(inputs));
            return Output.Select(o => o.Output).ToArray();
        }

        public void Learn(int maxEpochs = 10000, float rate = 0.01f, float errorLimit = 0.1f)
        {
            for (int epoch = 1; epoch <= maxEpochs; epoch++)
            {
                Loss = 0;
                foreach (var item in Data)
                {
                    var result = Predict(item.inputs);
                    for (int o = 0; o < Output.Length; o++)
                    {
                        //Calculating error
                        Loss += Neuron.Loss(result[o], item.target[o]) / Output.Length;

                        //Backpropagation
                        Output[o].Error = Neuron.dLoss(result[o], item.target[o]);
                        for (int l = Hidden.Length - 1; l >= 0; l--)
                            for (int n = 0; n < Hidden[0].Length; n++)
                                Hidden[l][n].Error = (l == Hidden.Length - 1) ? Output[o].Error * Output[o].Weight[n]
                                    : Hidden[l + 1][n].Error * Hidden[l + 1][n].Weight[n];

                        //Adjusting weights
                        Output[o].Adjust(rate);
                        Array.ForEach(Hidden, l => Array.ForEach(l, n => n.Adjust(rate)));
                    }
                }
                Loss /= Data.Count;
                Epochs = $"{epoch}/{maxEpochs}";
                if (Loss <= errorLimit || epoch >= maxEpochs) return;
            }
        }

        public void Save(string name)
        {
            var options = new JsonSerializerOptions
            {
                IgnoreReadOnlyProperties = true,
                IgnoreReadOnlyFields = true,
                WriteIndented = true
            };
            var data = JsonSerializer.Serialize<Network>(this, options);
            File.WriteAllText($"{name}.json", data);
        }

        public void Regularize(float offset = 0.01f)
        {
            Array.ForEach(Hidden, l => Array.ForEach(l, n => n.Regularize(offset)));
        }

        // https://csv2chart.com/
        public void Draw(string file)
        {
            var csv = new StringBuilder();
            for (int i = 0; i < 100; i++)
            {
                var y = i * 0.01f;
                var x = Predict(y)[0];
                csv.Append($"{y:n2} {x:n2}\n"
                    .Replace(",", ".")
                        .Replace(" ", ", "));
            }
            File.WriteAllText($"{file}.csv", csv.ToString());
        }

        public class Neuron
        {
            public enum ActivationType { None, Sigmoid, Tanh, ReLU, SoftPlus, ArcTan }
            public ActivationType Function { get; set; }
            public float Bias { get; set; } = 0;
            public float[] Weight { get; set; }
            public float[] Input;
            public float Error;
            public float Raw;
            public float Output;
            public float Derivative;

            public Neuron() { }

            public Neuron(int inputs, ActivationType function, float range = 2)
            {
                var Seed = new Random();
                var Rnd = (float min, float max)
                    => Seed.NextSingle() * Math.Abs(max - min) + min;
                Function = function;
                Weight = Enumerable.Range(0, inputs)
                    .Select(i => Rnd(-range, range)).ToArray();
            }

            public void Update(float[] input = null)
            {
                Input = input ?? Input;
                Raw = Input.Zip(Weight, (a, b) => a * b).Sum() + Bias;
                Output = Activation($"{Function}", Raw);
                Derivative = Activation($"d{Function}", Output);
            }

            public void Adjust(float rate = 0.01f)
            {
                Bias -= rate * Error * Derivative;
                for (int i = 0; i < Weight.Length; i++)
                    Weight[i] -= rate * Error * Derivative * Input[i];
                Error = 0;
            }

            public void Regularize(float offset = 0.01f)
            {
                Bias *= 1 - offset;
                for (int i = 0; i < Weight.Length; i++)
                    Weight[i] *= 1 - offset;
            }

            float Activation(string name, float x)
            {
                var method = this.GetType().GetMethod(name);
                return (float)method.Invoke(this, new object[] { x });
            }

            //Loss functions
            public static float Loss(float x, float target) => (float)Math.Sqrt(Math.Abs(target - x));
            public static float dLoss(float x, float target) => 2 * (x - target);

            //Activation functions
            public static float None(float x) => x;
            public static float dNone(float x) => 1;
            public static float ReLU(float x) => x <= 0f ? 0.01f * x : x;
            public static float dReLU(float x) => x < 0f ? 0f : 1f;
            public static float Sigmoid(float x) => (float)(1f / (1f + Math.Exp(-x)));
            public static float dSigmoid(float x) => Sigmoid(x) * (1f - Sigmoid(x));
            public static float SoftPlus(float x) => (float)Math.Log(1f + Math.Exp(x));
            public static float dSoftPlus(float x) => (float)(Math.Exp(x) / (1f + Math.Exp(x)));
            public static float Tanh(float x) => (float)Math.Tanh(x);
            public static float dTanh(float x) => (float)(1f - Math.Pow(Math.Tanh(x), 2));
            public static float ArcTan(float x) => (float)Math.Atan(x);
            public static float dArcTan(float x) => (float)(1f / (1f + Math.Pow(x, 2)));
        }
    }
}