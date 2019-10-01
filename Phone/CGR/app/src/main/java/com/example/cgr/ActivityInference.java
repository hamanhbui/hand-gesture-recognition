package com.example.cgr;
import android.content.Context;
import android.content.res.AssetManager;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

public class ActivityInference {
    private static ActivityInference activityInferenceInstance;
    private TensorFlowInferenceInterface inferenceInterface;
    private static final String MODEL_FILE = "file:///android_asset/frozen_mnist_convnet_gesture.pb";
    private static final String INPUT_NODE = "input";
    private static final String[] OUTPUT_NODES = {"output"};
    private static final String OUTPUT_NODE = "output";
    private static final int OUTPUT_SIZE = 24;
    private static AssetManager assetManager;

    public static ActivityInference getInstance(final Context context)
    {
        if (activityInferenceInstance == null)
        {
            activityInferenceInstance = new ActivityInference(context);
        }
        return activityInferenceInstance;
    }

    public ActivityInference(final Context context) {
        this.assetManager = context.getAssets();
        inferenceInterface = new TensorFlowInferenceInterface(assetManager, MODEL_FILE);
    }

    public float[] getActivityProb(float[] input_signal)
    {
        float[] result = new float[OUTPUT_SIZE];
        inferenceInterface.feed(INPUT_NODE,input_signal,1,50,6,1);
        inferenceInterface.run(OUTPUT_NODES);
        inferenceInterface.fetch(OUTPUT_NODE,result);
        return result;
    }
}
