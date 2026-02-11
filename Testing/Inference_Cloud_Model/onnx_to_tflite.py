import onnx
import onnx_tf.backend
import tensorflow as tf
import numpy as np
import onnxruntime as ort
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Step 1: Verify ONNX model
def verify_onnx_model(onnx_path):
    logger.info("Verifying ONNX model...")
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)  # Adjust shape if needed
    outputs = session.run(None, {input_name: dummy_input})
    logger.info(f"ONNX outputs: {[out.shape for out in outputs]}")
    return outputs

# Step 2: Convert ONNX to TensorFlow SavedModel
def onnx_to_tf(onnx_path, tf_path):
    logger.info("Converting ONNX to TensorFlow SavedModel...")
    onnx_model = onnx.load(onnx_path)
    tf_model = onnx_tf.backend.prepare(onnx_model)
    tf_model.export_graph(tf_path)
    logger.info(f"TensorFlow SavedModel saved to: {tf_path}")

# Step 3: Convert TensorFlow SavedModel to TFLite
def tf_to_tflite(tf_path, tflite_path, quantize="float32"):
    logger.info("Converting TensorFlow SavedModel to TFLite...")
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)
    
    if quantize == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        def representative_dataset_gen():
            # Replace with your cricket dataset (100-300 images)
            for _ in range(100):
                yield [np.random.randn(1, 3, 640, 640).astype(np.float32)]
        converter.representative_dataset = representative_dataset_gen
    elif quantize == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    else:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Dynamic range quantization

    tflite_model = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    logger.info(f"TFLite model saved to: {tflite_path}")

# Step 4: Verify TFLite model
def verify_tflite_model(tflite_path):
    logger.info("Verifying TFLite model...")
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_data = np.random.randn(1, 3, 640, 640).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    outputs = [interpreter.get_tensor(out['index']) for out in output_details]
    logger.info(f"TFLite outputs: {[out.shape for out in outputs]}")
    return outputs

def main():
    onnx_path = "best3.onnx"  # Update with your ONNX model path
    tf_path = "yolo11_tf_savedmodel"
    tflite_path = "best3.tflite"
    
    try:
        # Verify ONNX
        verify_onnx_model(onnx_path)
        
        # Convert to TensorFlow
        onnx_to_tf(onnx_path, tf_path)
        
        # Convert to TFLite (choose quantization: "float32", "float16", or "int8")
        tf_to_tflite(tf_path, tflite_path, quantize="int8")  # Use int8 for mobile
        
        # Verify TFLite
        verify_tflite_model(tflite_path)
        
    except Exception as e:
        logger.error(f"Error during conversion: {str(e)}")
        raise

if __name__ == "__main__":
    main()