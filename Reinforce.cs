public class Reinforce
{
    public float[,] QTable;
    int[] Actions;

    public Reinforce(int states, int actions)
    {
        QTable = new float[states, actions];
        Actions = Enumerable.Range(0, actions).ToArray();
        Experience = new (List<float>, int)[states, actions];
    }

    public static int MaxStates(int inputs, int inputRange = 3)
    {
        return (int)Math.Pow(inputRange, inputs);
    }

    public static int GetState(bool[] inputs)
    {
        return inputs.Select((input, i) => input ? (int)Math.Pow(2, i) : 0).Sum();
    }

    public static int GetState(int[] inputs, int inputRange = 3)
    {
        return inputs.Select((input, i) => input * (int)Math.Pow(inputRange, i)).Sum();
    }

    static Random Rnd = new Random();
    public int SelectAction(int state, float exploration = 0.1f)
    {
        if (Rnd.NextDouble() < exploration)
            return Rnd.Next(Actions.Length);
        else
            return Actions.MaxBy(i => QTable[state, i]);
    }

    public void Normal(int state, int action, float reward, float rate = 0.1f)
    {
        QTable[state, action] += rate * (reward - QTable[state, action]);
    }

    public void QLearning(int state, int action, int nextState, float reward, float rate = 0.1f, float discount = 0.9f)
    {
        var maxQ = Actions.Max(i => QTable[nextState, i]);
        QTable[state, action] += rate * (reward + discount * maxQ - QTable[state, action]);
    }

    public void SARSA(int state, int action, int nextState, int nextAction, float reward, float rate = 0.1f, float discount = 0.9f)
    {
        var nextQ = QTable[nextState, nextAction];
        QTable[state, action] += rate * (reward + discount * nextQ - QTable[state, action]);
    }

    public void ExpectedQ(int state, int action, int nextState, float reward, float rate = 0.1f, float discount = 0.9f)
    {
        var expectedQ = Actions.Average(i => QTable[nextState, i]);
        QTable[state, action] += rate * (reward + discount * expectedQ - QTable[state, action]);
    }

    (List<float> reward, int steps)[,] Experience;
    public void MonteCarlo(int state, int action, float reward, int maxSteps = 10, float rate = 0.1f)
    {
        var exp = Experience[state, action];
        exp.reward.Add(reward);
        if (exp.reward.Count > maxSteps)
            exp.reward.RemoveAt(0);

        if (exp.steps == maxSteps)
            QTable[state, action] += rate * (exp.reward.Average() - QTable[state, action]);

        exp.steps = exp.steps == maxSteps ? 0 : exp.steps + 1;
        Experience[state, action] = exp;
    }
}