

const net = new brain.recurrent.LSTMTimeStep({
    inputSize: 2,
    hiddenLayers: [10],
    outputSize: 2,
  });
  
  // Same test as previous, but combined on a single set
  const trainingData2 = [
    [
      [21-01-01, 21-05-01],
      [21-02-01, 21-04-01],
      [21-03-01, 21-03-01],
      [21-04-01, 21-02-01],
      [21-05-01, 21-01-01],
    ],
  ];
  
  net.train(trainingData2, { log: true, errorThresh: 0.09 });
  
  const closeToFiveAndOne = net.run([
    [21-01-01, 21-05-01],
    [21-02-01, 21-04-01],
    [21-03-01, 21-03-01],
    [21-04-01, 21-02-01],
  ]);
  
  console.log(closeToFiveAndOne);
  
  // now we're cookin' with gas!
  const forecast = net.forecast(
    [
      [21-01-01, 21-05-01],
      [21-02-01, 21-04-01],
    ],
    3
  );
  
  console.log('next 3 predictions', forecast);