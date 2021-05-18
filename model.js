import * as tf from "@tensorflow/tfjs-node";
import * as learnTodos from './data/learn_todos.js';
import * as exerciseTodos from './data/exercise_todos.js';
import use from "@tensorflow-models/universal-sentence-encoder";
import * as gdpCategories from './data/gdp-categories.js';
// const trainTasks = learnTodos.data.concat(exerciseTodos.data);
const text = 'The waiter was rude';

const trainTasks = gdpCategories.data;
const MODEL_NAME = "suggestion-model";
const N_CLASSES = 3;

const encodeData = async (encoder, tasks) => {
  const sentences = tasks.map(t => t.text.toLowerCase());
  const embeddings = await encoder.embed(sentences);
  return embeddings;
};

const trainModel = async () => {
  const encoder = await use.load();
  try {
    const handler = tf.io.fileSystem("./model-1a/model.json");
    const model = await tf.loadLayersModel(handler);
    console.log("Using existing model");
    return model;
  } catch (e) {
    console.log(e)
    console.log("Training new model");
    const xTrain = await encodeData(encoder, trainTasks);
    const yTrain = tf.tensor2d(
      trainTasks.map(t => [t.category === "Food Quality" ? 1 : 0, t.category === "Service" ? 1 : 0, t.category === "Ambiance" ? 1 : 0])
    );
    const model = tf.sequential();
    model.add(
      tf.layers.dense({
        inputShape: [xTrain.shape[1]],
        activation: "softmax",
        units: N_CLASSES
      })
    );
    model.compile({
      loss: "categoricalCrossentropy",
      optimizer: tf.train.adam(0.001),
      metrics: ["accuracy"]
    });
    await model.fit(xTrain, yTrain, {
      batchSize: 32,
      validationSplit: 0.1,
      shuffle: true,
      epochs: 185
    });
    const saved = await model.save("file://./model-1a");
    return model;
  }
};

const suggestIcon = async (taskName, threshold=0.65) => {
  if (!taskName.trim().includes(" ")) {
    return null;
  }
  const encoder = await use.load();
  const model = await trainModel();
  const xPredict = await encodeData(encoder, [{ text: taskName }]);

  const prediction = await model.predict(xPredict).data();
  console.log('prediction', prediction)

  if (prediction[0] > threshold) {
    return "Food Quality";
  } else if (prediction[1] > threshold) {
    return "Service";
  }else if(prediction[2]>threshold){
    return 'Ambiance'
  } else {
    return null;
  }
};
(async ()=>{
  const data = await suggestIcon(text);
  console.log("text>>>>>>>>>", text)
  console.log('Category', data)
})();
// trainModel();
