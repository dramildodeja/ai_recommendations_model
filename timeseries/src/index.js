import "./styles.css";
import Plotly from "plotly.js-dist";
import * as tf from "@tensorflow/tfjs";
import * as tfvis from "@tensorflow/tfjs-vis";
import * as Papa from "papaparse";
import _ from "lodash";
import { runPrediction } from "./prediction.js";
const predictionOutputSection = document.getElementById("predicton_output_section");

//converting set of sparse labels to a dense one-hot representation
//2 because we have 2 outputs in the prediction results...Customer or No-Customer
const oneHot = (outcome) => Array.from(tf.oneHot(outcome, 2).dataSync());

//this is a lib for efficiently parsing csv data for preparing datasets
Papa.parsePromise = function (file) {
  return new Promise(function (complete, error) {Papa.parse(file, {header: true,download: true,dynamicTyping: true,complete,error});});
};
const prepareData = async () => {
  const csv = await Papa.parsePromise("https://raw.githubusercontent.com/dramildodeja/ai_recommendations_model/main/timeseries/src/data/training_data_1.csv");
  return csv.data;
};

//creating datasets prior to train the model from the raw data
const createDataSets = (data, features, testSize, batchSize) => {
  const X = data.map((r) =>
    features.map((f) => {
      const val = r[f];
      return val === undefined ? 0 : val;
    }),
  );
  // The outcome of the model with one-hot encoding to represent the outcome
  //regression on client_converted field...Customer or No-Customer aka 0 or 1
  const y = data.map((r) => {
    const outcome = r.client_converted === undefined ? 0 : r.client_converted;
    return oneHot(outcome);
  });

  //Split the data into training and testing sets
  //Why this is important - https://builtin.com/data-science/train-test-split
  const splitIdx = parseInt((1 - testSize) * data.length, 10);
  // Create a dataset from the data
  // zip the features and outcome together then shuffle the data and split it into batches (size 42)
  const ds = tf.data.zip({ xs: tf.data.array(X), ys: tf.data.array(y)}).shuffle(data.length, 42);
  return [
    ds.take(splitIdx).batch(batchSize),
    ds.skip(splitIdx + 1).batch(batchSize),
    tf.tensor(X.slice(splitIdx)),
    tf.tensor(y.slice(splitIdx)),
  ];
};

const trainMyLTSMModel = async (featureCount, trainDs, validDs) => {
  const model = tf.sequential();

  // Add a dense layer
  // Dense layers are fully connected layers
  // units: 2, because we have 2 outcomes
  // activation: softmax, because we want to classify the data
  // softmax is a function that squashes the values between 0 and 1
  // inputShape: [featureCount], because we have featureCount number of features
  model.add(
    tf.layers.dense({
      units: 2,
      activation: "softmax",
      inputShape: [featureCount],
    }),
  );

  // Compile the model
  // we use adam optimizer that is a popular optimizer
  // loss: binaryCrossentropy, because we have 2 outcomes
  // we want to minimize the loss
  const optimizer = tf.train.adam(0.001);
  model.compile({
    optimizer: optimizer,
    loss: "binaryCrossentropy",
    metrics: ["accuracy"],
  });

  // Train the model
  const trainLogs = [];
  const lossContainer = document.getElementById("loss-cont");
  const accContainer = document.getElementById("acc-cont");
  //console.log("Training...");
  // We train the model for 100 epochs meaning we go through the dataset 100 times
  // We also pass the validation data so we can see how the model performs on unseen data
  // We use callbacks to log the loss and accuracy
  // We also use tfvis to visualize the training process
  await model.fitDataset(trainDs, {
    epochs: 100,
    validationData: validDs,
    callbacks: {
      onEpochEnd: async (_, logs) => {
        trainLogs.push(logs);
        tfvis.show.history(lossContainer, trainLogs, ["loss", "val_loss"]);
        tfvis.show.history(accContainer, trainLogs, ["acc", "val_acc"]);
      },
    },
  });
  //console.log("Model training completed");
  return model;
};

const renderOutcomes = (data) => {
  const outcomes = data.map((r) => r.client_converted);
  const [customer, noCustomer] = _.partition(outcomes, (o) => o === 1);
  const chartData = [
  {
    labels: ["Customer", "NoCustomer"],
    values: [customer.length, noCustomer.length],
    type: "pie",
    opacity: 0.6,
    marker: { colors: ["green", "red"],
    },
  }];
  Plotly.newPlot("predicton_output_section", chartData, { title: "Customer vs NoCustomer"});
};

//Init function
const run = async () => {
  const data = await prepareData();
  renderOutcomes(data);
  const features = ["client_presentation_completed", "revisiting_lead_status"];
  const [trainDs, validDs, xTest, yTest] = createDataSets(
    data,
    features,
    0.1,
    16,
  );
  const model = await trainMyLTSMModel(
    features.length,//data columns
    trainDs,
    validDs,
  );
  return model;
};

if (document.readyState !== "loading") {
  run().then((model) => {
    predictionOutputSection.style.display = "flex";
    runPrediction(model);
  });
} else {
  document.addEventListener("DOMContentLoaded", () => {
    run().then((model) => {
      predictionOutputSection.style.display = "flex";
      runPrediction(model);
    });
  });
}
