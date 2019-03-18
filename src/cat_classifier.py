import cv2
import numpy as np
import tensorflow as tf


class CatClassifier(object):
    def __init__(self, model_path):
        self._graph = self._load_model(model_path)
        self._sess = tf.Session(graph=self._graph)
        self._input_image = self._graph.get_tensor_by_name(
            'cat_classifier/input_image:0'
        )
        self._logits = self._graph.get_tensor_by_name(
            'cat_classifier/logits:0'
        )

    def _load_model(self, model_path):
        with tf.gfile.GFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def,
                                input_map=None,
                                return_elements=None,
                                name='cat_classifier',
                                producer_op_list=None)
        return graph

    def _softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def _assign_label(self, p):
        if p[1] > 0.9999:
            return True
        else:
            return False

    def predict(self, image, bboxes=None):
        if bboxes:
            images = [
                cv2.resize(
                    crop_image(
                        image,
                        bbox),
                    (128,
                     128),
                    interpolation=cv2.INTER_AREA) for bbox in bboxes]
        elif not bboxes:
            images = [
                cv2.resize(
                    cat, (128, 128), interpolation=cv2.INTER_AREA) for cat in image]

        images = [image / 255.0 for image in images]

        prediction = self._sess.run(self._logits,
                                    feed_dict={self._input_image: images})

        probs = [self._softmax(logit) for logit in prediction]

        results = [self._assign_label(p) for p in probs]
        probs = [prob[np.argmax(prob)] for prob in probs]
        return results, probs


if __name__ == '__main__':
    c = CatClassifier("./cat_model.pb")
    image = cv2.imread("./cat7.jpg")
    print(c.predict([image]))
