# Waste-Classification-Assistant
Waste Classification Assistant

A prototype for image classification built using transfer learning with MobileNetV2. Given a photo of a single item of waste, the system identifies which of six categories it belongs to—cardboard, glass, metal, paper, plastic, or trash—and provides a confidence score. Inference is performed using TensorFlow and Keras, with a Flask web interface for demonstration.

Week 1: Data pipeline and frozen-base baseline.
Week 2: Fine-tuning, class weighting, and ablation studies.
Week 3: Inference scripts and web-based demonstration.

Results

On a held-out test set of 379 images, the frozen-base baseline achieved 80.2% accuracy; fine-tuning also yielded 80.2%, while the addition of class weights resulted in 79.2%.
No significant differences were observed between the three models. With 379 images and an accuracy of approximately 80%, the standard error is about 2.1 percentage points, and the 95% confidence interval is roughly ±4 percentage points; variations from re-running the same code are greater than the differences between the models themselves.
The only effect that clearly stands out above the noise is the use of transfer learning: a CNN trained from scratch achieves only 22% accuracy (Yang & Thung, 2016), whereas this approach reaches approximately 80%.

Environment

Python 3.10, TensorFlow 2.15.1 with Keras 2.15; additional dependencies include NumPy, scikit-learn, Matplotlib, Pillow, and Flask.
