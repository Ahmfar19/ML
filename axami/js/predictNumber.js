

const net = new brain.recurrent.LSTMTimeStep({
  inputSize: 2,
  hiddenLayers: [10],
  outputSize: 2,
});

// Same test as previous, but combined on a single set
const trainingData2 = [
  [
    [1, 5],
    [2, 4],
    [3, 3],
    [4, 2],
    [5, 1],
  ],
];

net.train(trainingData2, { log: true, errorThresh: 0.09 });

const closeToFiveAndOne = net.run([
  [1, 5],
  [2, 4],
  [3, 3],
  [4, 2],
]);

console.log(closeToFiveAndOne);

// now we're cookin' with gas!
const forecast = net.forecast(
  [
    [1, 5],
    [2, 4],
  ],
  15
);

console.log('next 3 predictions', forecast);