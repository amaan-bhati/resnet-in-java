import java.util.Random;

public class ResNet {

    public static double[][][] convolution2d(double[][][] inputTensor, int filters, int[] kernelSize, int[] strides, String padding) {
        int inputHeight = inputTensor.length;
        int inputWidth = inputTensor[0].length;
        int inputChannels = inputTensor[0][0].length;
        int filterHeight = kernelSize[0];
        int filterWidth = kernelSize[1];
        int strideHeight = strides[0];
        int strideWidth = strides[1];

        int padHeight = 0, padWidth = 0;
        if (padding.equals("same")) {
            padHeight = ((inputHeight - 1) * strideHeight + filterHeight - inputHeight) / 2;
            padWidth = ((inputWidth - 1) * strideWidth + filterWidth - inputWidth) / 2;
            inputTensor = padTensor(inputTensor, padHeight, padWidth);
        }

        int outputHeight = (inputHeight + 2 * padHeight - filterHeight) / strideHeight + 1;
        int outputWidth = (inputWidth + 2 * padWidth - filterWidth) / strideWidth + 1;

        double[][][] outputTensor = new double[outputHeight][outputWidth][filters];
        Random random = new Random();

        for (int f = 0; f < filters; f++) {
            for (int i = 0; i <= inputHeight - filterHeight; i += strideHeight) {
                for (int j = 0; j <= inputWidth - filterWidth; j += strideWidth) {
                    double sum = 0;
                    for (int fi = 0; fi < filterHeight; fi++) {
                        for (int fj = 0; fj < filterWidth; fj++) {
                            for (int fc = 0; fc < inputChannels; fc++) {
                                sum += inputTensor[i + fi][j + fj][fc] * random.nextDouble();
                            }
                        }
                    }
                    outputTensor[i / strideHeight][j / strideWidth][f] = sum;
                }
            }
        }
        return outputTensor;
    }

    public static double[][][] padTensor(double[][][] inputTensor, int padHeight, int padWidth) {
        int inputHeight = inputTensor.length;
        int inputWidth = inputTensor[0].length;
        int inputChannels = inputTensor[0][0].length;

        double[][][] paddedTensor = new double[inputHeight + 2 * padHeight][inputWidth + 2 * padWidth][inputChannels];

        for (int i = 0; i < inputHeight; i++) {
            for (int j = 0; j < inputWidth; j++) {
                for (int k = 0; k < inputChannels; k++) {
                    paddedTensor[i + padHeight][j + padWidth][k] = inputTensor[i][j][k];
                }
            }
        }
        return paddedTensor;
    }

    public static double[][][] batchNormalization(double[][][] inputTensor) {
        int height = inputTensor.length;
        int width = inputTensor[0].length;
        int channels = inputTensor[0][0].length;

        double[] mean = new double[channels];
        double[] variance = new double[channels];

        for (int c = 0; c < channels; c++) {
            double sum = 0;
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    sum += inputTensor[i][j][c];
                }
            }
            mean[c] = sum / (height * width);
        }

        for (int c = 0; c < channels; c++) {
            double sum = 0;
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    sum += Math.pow(inputTensor[i][j][c] - mean[c], 2);
                }
            }
            variance[c] = sum / (height * width);
        }

        double epsilon = 1e-5;
        double[][][] normalizedTensor = new double[height][width][channels];

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                for (int c = 0; c < channels; c++) {
                    normalizedTensor[i][j][c] = (inputTensor[i][j][c] - mean[c]) / Math.sqrt(variance[c] + epsilon);
                }
            }
        }
        return normalizedTensor;
    }

    public static double[][][] relu(double[][][] inputTensor) {
        int height = inputTensor.length;
        int width = inputTensor[0].length;
        int channels = inputTensor[0][0].length;

        double[][][] outputTensor = new double[height][width][channels];

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                for (int c = 0; c < channels; c++) {
                    outputTensor[i][j][c] = Math.max(0, inputTensor[i][j][c]);
                }
            }
        }
        return outputTensor;
    }

    public static double[][][] resnetBlock(double[][][] inputTensor, int filters, int[] kernelSize, int[] strides, boolean useProjection) {
        double[][][] shortcut = inputTensor;

        double[][][] x = convolution2d(inputTensor, filters, kernelSize, strides, "same");
        x = batchNormalization(x);
        x = relu(x);

        x = convolution2d(x, filters, kernelSize, new int[]{1, 1}, "same");
        x = batchNormalization(x);

        if (useProjection) {
            shortcut = convolution2d(shortcut, filters, new int[]{1, 1}, strides, "same");
            shortcut = batchNormalization(shortcut);
        }

        x = addTensors(x, shortcut);
        x = relu(x);

        return x;
    }

    public static double[][][] addTensors(double[][][] tensor1, double[][][] tensor2) {
        int height = tensor1.length;
        int width = tensor1[0].length;
        int channels = tensor1[0][0].length;

        double[][][] outputTensor = new double[height][width][channels];

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                for (int c = 0; c < channels; c++) {
                    outputTensor[i][j][c] = tensor1[i][j][c] + tensor2[i][j][c];
                }
            }
        }
        return outputTensor;
    }

    public static double[] globalAveragePooling2d(double[][][] inputTensor) {
        int height = inputTensor.length;
        int width = inputTensor[0].length;
        int channels = inputTensor[0][0].length;

        double[] outputTensor = new double[channels];

        for (int c = 0; c < channels; c++) {
            double sum = 0;
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    sum += inputTensor[i][j][c];
                }
            }
            outputTensor[c] = sum / (height * width);
        }
        return outputTensor;
    }

    public static double[] fullyConnectedLayer(double[] inputTensor, double[][] weights, double[] biases) {
        int numOutputs = biases.length;
        double[] outputTensor = new double[numOutputs];

        for (int i = 0; i < numOutputs; i++) {
            double sum = 0;
            for (int j = 0; j < inputTensor.length; j++) {
                sum += inputTensor[j] * weights[j][i];
            }
            outputTensor[i] = sum + biases[i];
        }
        return outputTensor;
    }

    public static double[] softmaxActivation(double[] inputVector) {
        double[] expValues = new double[inputVector.length];
        double sum = 0;

        for (int i = 0; i < inputVector.length; i++) {
            expValues[i] = Math.exp(inputVector[i]);
            sum += expValues[i];
        }

        for (int i = 0; i < inputVector.length; i++) {
            expValues[i] /= sum;
        }

        return expValues;
    }

    public static double[] buildResNet(int[] inputShape, int numClasses) {
        Random random = new Random();
        double[][][] inputTensor = new double[inputShape[0]][inputShape[1]][inputShape[2]];
        for (int i = 0; i < inputShape[0]; i++) {
            for (int j = 0; j < inputShape[1]; j++) {
                for (int k = 0; k < inputShape[2]; k++) {
                    inputTensor[i][j][k] = random.nextDouble();
                }
            }
        }

        double[][][] x = convolution2d(inputTensor, 64, new int[]{7, 7}, new int[]{2, 2}, "same");
        x = batchNormalization(x);
        x = relu(x);

        x = resnetBlock(x, 64, new int[]{3, 3}, new int[]{1, 1}, false);
        x = resnetBlock(x, 64, new int[]{3, 3}, new int[]{1, 1}, false);
        x = resnetBlock(x, 128, new int[]{3, 3}, new int[]{2, 2}, true);
        x = resnetBlock(x, 128, new int[]{3, 3}, new int[]{1, 1}, false);
        x = resnetBlock(x, 256, new int[]{3, 3}, new int[]{2, 2}, true);
        x = resnetBlock(x, 256, new int[]{3, 3}, new int[]{1, 1}, false);

        double[] pooled = globalAveragePooling2d(x);

        double[][] weights = new double[pooled.length][numClasses];
        double[] biases = new double[numClasses];
        for (int i = 0; i < pooled.length; i++) {
            for (int j = 0; j < numClasses; j++) {
                weights[i][j] = random.nextDouble();
            }
        }
        for (int i = 0; i < numClasses; i++) {
            biases[i] = random.nextDouble();
        }

        double[] output = fullyConnectedLayer(pooled, weights, biases);
        output = softmaxActivation(output);

        return output;
    }

    public static void main(String[] args) {
        int[] inputShape = {32, 32, 3};
        int numClasses = 10;

        double[] resNetOutput = buildResNet(inputShape, numClasses);

        System.out.println("ResNet Output:");
        for (double value : resNetOutput) {
            System.out.print(value + " ");
        }
    }
}
