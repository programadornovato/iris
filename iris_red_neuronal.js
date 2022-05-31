function getIrisData(testSplit) {
	return tf.tidy(() => {
		const dataByClass = [];
		const targetByClass = [];
		for (let i = 0; i < IRIS_CLASSES.length; i++) {
			dataByClass.push([]);
			targetByClass.push([]);
		}
		for (const example of IRIS_DATA) {
			const target=example[example.length-1];
			const data=example.slice(0,example.length-1);
			dataByClass[target].push(data);
			targetByClass[target].push(target);
		}
		const xTrains = [];
		const yTrains = [];
		const xTests = [];
		const yTests = [];
		for (let i = 0; i < IRIS_CLASSES.length; i++) {
			const [xTrain, yTrain, xTest, yTest] = convertToTensors(dataByClass[i], targetByClass[i], testSplit);
			xTrains.push(xTrain);
			yTrains.push(yTrain);
			xTests.push(xTest);
			yTests.push(yTest);
		}
		const concatAxis = 0;
		const test1=xTrains;
		const test2=tf.concat(xTrains, concatAxis);
		console.log(test1);
		console.log(test2);
		return [tf.concat(xTrains, concatAxis), tf.concat(yTrains, concatAxis),
			tf.concat(xTests, concatAxis), tf.concat(yTests, concatAxis)
		];
	});
}
function convertToTensors(data, target, testSplit) {
	const numExamples = data.length;
	if(numExamples!==target.length){
		throw new Error('data and target must be the same length');
	}
	const numTestExamples = Math.round(numExamples * testSplit);
	const numTrainExamples = numExamples - numTestExamples;
	const xDims = data[0].length;
	const xs = tf.tensor2d(data, [numExamples, xDims]);
	const ys = tf.oneHot(tf.tensor1d(target).toInt(), IRIS_NUM_CLASSES);
	const xTrain = xs.slice([0, 0], [numTrainExamples, xDims]);
	const yTrain = ys.slice([0, 0], [numTrainExamples, IRIS_NUM_CLASSES]);
	const xTest = xs.slice([numTrainExamples, 0], [numTestExamples, xDims]);
	const yTest = ys.slice([numTrainExamples, 0], [numTestExamples, IRIS_NUM_CLASSES]);
	return [xTrain, yTrain, xTest, yTest];
}