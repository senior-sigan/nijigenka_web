const imageSize = 256;
const modelPath = "./twingan_js.onnx";

async function getSession() {
  if (window.session) {
    return window.session;
  }

  const session = new onnx.InferenceSession({ backendHint: 'webgl' });
  await session.loadModel(modelPath);
  
  window.session = session;
  return session;
}

async function animefaction(imageUrl) {
  showMessage('Loading image...');
  const imageLoader = new ImageLoader(imageSize, imageSize);
  const imageData = await imageLoader.getImageData(imageUrl);
  
  drawInput(imageData.data, imageSize, imageSize);
  const outputImage = putOutputPlaceholder();

  showMessage('Preparing image...');
  const preprocessedData = preprocess(imageData.data, imageSize, imageSize);

  showMessage('Loading model...');
  const session = await getSession();

  showMessage('Animefication!...');
  const inputTensor = new onnx.Tensor(preprocessedData, 'float32', [1, imageSize, imageSize, 3]);
  const outputMap = await session.run([inputTensor]);
  const outputData = outputMap.values().next().value.data;

  showMessage('Postprocessing...');
  const processesOutput = postprocess(outputData, imageSize, imageSize);
  const outputImageData = toImageDataURL(processesOutput, imageSize, imageSize);
  outputImage.src = outputImageData;
  showMessage('Success! Select next selfie.');
}

function preprocess(data, width, height) {
  const dataFromImage = ndarray(new Float32Array(data), [height, width, 4]);
  const dataProcessed = ndarray(new Float32Array(width * height * 3), [1, height, width, 3]);

  // Normalize 0-255 to (-1)-1
  ndarray.ops.divseq(dataFromImage, 256.0);

  // Realign imageData from [H*W*4] to the correct dimension [1*H*W*3].
  ndarray.ops.assign(dataProcessed.pick(0, null, null, 0), dataFromImage.pick(null, null, 0));
  ndarray.ops.assign(dataProcessed.pick(0, null, null, 1), dataFromImage.pick(null, null, 1));
  ndarray.ops.assign(dataProcessed.pick(0, null, null, 2), dataFromImage.pick(null, null, 2));

  return dataProcessed.data;
}

function postprocess(data, width, height) {
  const dataRaw = ndarray(new Float32Array(data), [1, height, width, 3]);
  const dataImage = ndarray(new Uint8ClampedArray(width * height * 4), [height, width, 4]);
  dataImage.data.fill(255);

  // form 0..1 to 0..255
  ndarray.ops.mulseq(dataRaw, 256.0);

  // from [H*W*3] to [H*W*4]
  ndarray.ops.assign(dataImage.pick(null, null, 2), dataRaw.pick(0, null, null, 2));
  ndarray.ops.assign(dataImage.pick(null, null, 1), dataRaw.pick(0, null, null, 1));
  ndarray.ops.assign(dataImage.pick(null, null, 0), dataRaw.pick(0, null, null, 0));

  return dataImage.data;
}

function drawInput(imageData, width, height) {
  const image = new Image();
  image.src = toImageDataURL(imageData, width, height);
  document.body.appendChild(image);
}

function putOutputPlaceholder() {
  const image = new Image();
  image.src = 'spinner.gif';
  document.body.appendChild(image);
  return image;
}

function toImageDataURL(buffer, width, height) {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  canvas.width = width;
  canvas.height = height;
  const idata = ctx.createImageData(width, height);
  idata.data.set(buffer);
  ctx.putImageData(idata, 0, 0);
  return canvas.toDataURL();
}

function showMessage(text) {
  console.log(text);
  const messageArea = document.getElementById('message_area');
  message_area.textContent = text;
}