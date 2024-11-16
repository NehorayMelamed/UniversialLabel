import numpy as np
from typing import Dict, List, Tuple
import torch
import cv2
import os

class NMSHandler:
    """
    A class responsible for performing Non-Maximum Suppression (NMS) and prioritization across multiple models and classes.
    """
    def __init__(self):
        self.class_priorities = {}
        self.model_priorities = {}
        self.iou_threshold = 0.5  # Default IoU threshold for NMS

    def set_class_priorities(self, priorities: Dict[str, int]):
        """
        Set priorities for classes.
        Args:
            priorities (Dict[str, int]): Dictionary of class priorities. Higher value indicates higher priority.
        """
        self.class_priorities = priorities

    def set_model_priorities(self, priorities: Dict[str, int]):
        """
        Set priorities for models.
        Args:
            priorities (Dict[str, int]): Dictionary of model priorities. Higher value indicates higher priority.
        """
        self.model_priorities = priorities

    def set_iou_threshold(self, threshold: float):
        """
        Set the IoU threshold for NMS.
        Args:
            threshold (float): IoU threshold value.
        """
        self.iou_threshold = threshold

    def nms(self, boxes: np.ndarray, scores: np.ndarray) -> List[int]:
        """
        Perform Non-Maximum Suppression (NMS) on the bounding boxes.

        Args:
            boxes (np.ndarray): Bounding boxes, array of shape (N, 4).
            scores (np.ndarray): Confidence scores, array of shape (N,).

        Returns:
            List[int]: Indices of boxes kept after applying NMS.
        """
        if len(boxes) == 0:
            return []
        # Using torchvision.ops.nms
        from torchvision.ops import nms
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        scores_tensor = torch.tensor(scores, dtype=torch.float32)
        keep_indices = nms(boxes_tensor, scores_tensor, self.iou_threshold)
        return keep_indices.cpu().numpy().tolist()

    def merge_results(self, results: Dict[str, Dict[str, List]]) -> Dict[str, List]:
        """
        Merge detection results from multiple models and apply NMS with prioritization.

        Args:
            results (Dict[str, Dict[str, List]]): Dictionary containing detection results for each model.

        Returns:
            Dict[str, List]: A dictionary containing merged bounding boxes, labels, and scores.
        """
        combined_bboxes = []
        combined_labels = []
        combined_scores = []
        combined_model_names = []

        # Combine all results
        for model_name, result in results.items():
            combined_bboxes.extend(result.get('bboxes', []))
            combined_labels.extend(result.get('labels', []))
            combined_scores.extend(result.get('scores', []))
            combined_model_names.extend([model_name] * len(result.get('bboxes', [])))

        # Convert lists to numpy arrays for easier manipulation
        bboxes = np.array(combined_bboxes)
        scores = np.array(combined_scores)
        labels = np.array(combined_labels)
        model_names = np.array(combined_model_names)

        # Sort by model and class priorities
        priority_scores = []
        for label, model_name in zip(labels, model_names):
            class_priority = self.class_priorities.get(label, 0)
            model_priority = self.model_priorities.get(model_name, 0)
            priority_scores.append(class_priority + model_priority)

        # Sort the boxes by priority score (higher is better)
        sorted_indices = np.argsort(priority_scores)[::-1]
        bboxes = bboxes[sorted_indices]
        scores = scores[sorted_indices]
        labels = labels[sorted_indices]

        # Apply NMS
        if len(bboxes) > 0:
            keep_indices = self.nms(bboxes, scores)
            bboxes = bboxes[keep_indices]
            scores = scores[keep_indices]
            labels = labels[keep_indices]

        # Return a complete dictionary structure
        return {
            'bboxes': bboxes.tolist() if len(bboxes) > 0 else [],
            'labels': labels.tolist() if len(labels) > 0 else [],
            'scores': scores.tolist() if len(scores) > 0 else []
        }
    def apply_prioritization(self, bboxes: np.ndarray, labels: np.ndarray, scores: np.ndarray, model_names: np.ndarray) -> Tuple[List, List, List]:
        """
        Apply class-based and model-based prioritization to the bounding boxes.

        Args:
            bboxes (np.ndarray): Bounding boxes.
            labels (np.ndarray): Class labels.
            scores (np.ndarray): Confidence scores.
            model_names (np.ndarray): Model names corresponding to each bounding box.

        Returns:
            Tuple[List, List, List]: Filtered bounding boxes, labels, and scores after applying prioritization.
        """
        final_bboxes = []
        final_labels = []
        final_scores = []

        # Iterate through all bounding boxes and apply prioritization
        for i in range(len(bboxes)):
            bbox = bboxes[i]
            label = labels[i]
            score = scores[i]
            model_name = model_names[i]

            # Apply class priority if overlap occurs
            for j in range(i + 1, len(bboxes)):
                iou = self.calculate_iou(bbox, bboxes[j])
                if iou > self.iou_threshold:
                    if label in self.class_priorities and labels[j] in self.class_priorities:
                        if self.class_priorities[labels[j]] > self.class_priorities[label]:
                            label = labels[j]
                            score = scores[j]

            # Apply model priority
            if model_name in self.model_priorities:
                # Prioritize models with higher rank if overlapping
                for j in range(i + 1, len(bboxes)):
                    iou = self.calculate_iou(bbox, bboxes[j])
                    if iou > self.iou_threshold:
                        if self.model_priorities[model_names[j]] > self.model_priorities[model_name]:
                            model_name = model_names[j]
                            score = scores[j]

            final_bboxes.append(bbox)
            final_labels.append(label)
            final_scores.append(score)

        return final_bboxes, final_labels, final_scores

    @staticmethod
    def calculate_iou(boxA: np.ndarray, boxB: np.ndarray) -> float:
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.

        Args:
            boxA (np.ndarray): First bounding box in [x_min, y_min, x_max, y_max] format.
            boxB (np.ndarray): Second bounding box in [x_min, y_min, x_max, y_max] format.

        Returns:
            float: IoU value.
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def save_result(self, image: np.ndarray, bboxes: List[List[int]], labels: List[str], scores: List[float], output_path: str):
        """
        Save the detection result to the specified output path.

        Args:
            image (np.ndarray): The input image as a NumPy array.
            bboxes (List[List[int]]): List of bounding boxes in [x_min, y_min, x_max, y_max] format.
            labels (List[str]): List of class labels corresponding to each bounding box.
            scores (List[float]): List of confidence scores corresponding to each bounding box.
            output_path (str): The path to save the result image.
        """
        output_image = image.copy()
        for bbox, label, score in zip(bboxes, labels, scores):
            x_min, y_min, x_max, y_max = map(int, bbox)
            cv2.rectangle(output_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(output_image, f"{label} ({score:.2f})", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, output_image)
        print(f"Detection result saved to {output_path}")


# Example usage
if __name__ == "__main__":
    # Sample detection results from multiple models
    formatted_result = {
        "dino": {'bboxes': [[48, 90, 125, 173], [310, 135, 412, 255], [166, 68, 209, 127], [467, 159, 514, 198], [61, 167, 80, 196], [40, 219, 112, 258], [197, 123, 236, 153], [372, 107, 458, 205], [110, 74, 166, 142], [288, 94, 345, 173], [299, 75, 328, 91], [179, 144, 222, 179], [160, 235, 227, 257], [71, 183, 135, 219], [214, 80, 241, 97], [353, 66, 388, 91], [158, 209, 227, 251], [100, 155, 149, 203], [370, 89, 394, 108], [207, 89, 229, 111], [254, 69, 260, 78], [117, 230, 148, 257], [143, 61, 175, 76], [78, 204, 134, 241], [227, 74, 251, 93], [356, 80, 379, 101], [328, 78, 351, 93], [414, 104, 440, 116], [281, 60, 316, 88], [332, 83, 355, 104], [152, 71, 172, 80], [362, 105, 389, 136], [0, 159, 8, 257], [160, 170, 222, 213], [86, 174, 128, 193], [160, 191, 226, 229]],
                 'labels': ['bus', 'bus', 'bus', 'car', 'bike', 'car', 'car', 'bus', 'bus', 'bus', 'car', 'car', 'car', 'car', 'car', 'truck', 'car', 'car', 'car', 'car', 'bike', 'bike', 'bus', 'car', 'car', 'car', 'car', 'car', 'bus truck', 'car', 'car', 'car', 'bus truck', 'car', 'car', 'car'],
                 'scores': [0.6372516751289368, 0.6554409861564636, 0.5495069622993469, 0.7379211783409119, 0.7333720922470093, 0.49647191166877747, 0.5644051432609558, 0.6085228323936462, 0.5383757948875427, 0.5773429870605469, 0.49854400753974915, 0.47545528411865234, 0.4723787307739258, 0.4703662097454071, 0.4402005672454834, 0.4256379008293152, 0.4152977168560028, 0.3917722702026367, 0.4428081512451172, 0.5260006785392761, 0.4721960425376892, 0.5910893082618713, 0.3343741297721863, 0.427364319562912, 0.39918893575668335, 0.4158951938152313, 0.40467405319213867, 0.45588234066963196, 0.35159537196159363, 0.4241691827774048, 0.3207804262638092, 0.38378748297691345, 0.36908817291259766, 0.3329097628593445, 0.3196355402469635, 0.3680365979671478]},
        "yolo_world": {'bboxes': [[309.95013427734375, 135.1962890625, 413.50848388671875, 255.84869384765625], [48.29264831542969, 90.34847259521484, 126.27942657470703, 177.08590698242188], [164.7147979736328, 68.87649536132812, 209.1956024169922, 127.5923080444336], [467.99432373046875, 159.6128387451172, 514.5567626953125, 199.16485595703125], [374.0268249511719, 105.90724182128906, 458.5685729980469, 205.79580688476562], [110.2816390991211, 75.92340850830078, 166.74081420898438, 143.43966674804688], [44.12045669555664, 218.96754455566406, 112.93115997314453, 257.71954345703125], [290.5379638671875, 93.65755462646484, 346.76910400390625, 173.2947540283203], [198.14158630371094, 123.71001434326172, 235.31080627441406, 153.4168701171875], [179.96127319335938, 145.083251953125, 221.7531280517578, 182.10804748535156], [299.5683288574219, 75.7127456665039, 329.1260986328125, 92.0250244140625], [352.6388244628906, 66.00310516357422, 388.1128845214844, 101.438720703125], [100.13765716552734, 155.32980346679688, 149.51986694335938, 204.7285919189453], [207.8965606689453, 89.57101440429688, 228.5675811767578, 111.27497863769531], [61.709266662597656, 169.94534301757812, 78.30052947998047, 196.9456024169922], [369.8778076171875, 89.52264404296875, 394.015869140625, 107.46061706542969], [165.98863220214844, 236.15370178222656, 225.52223205566406, 258.0], [235.2001953125, 75.01757049560547, 251.8009033203125, 93.66548919677734], [281.4281311035156, 60.54228973388672, 316.238037109375, 88.52204895019531], [352.6760559082031, 66.28706359863281, 387.8790283203125, 90.27045440673828]],
                       'labels': ['bus', 'bus', 'bus', 'car', 'bus', 'bus', 'car', 'bus', 'car', 'car', 'car', 'truck', 'car', 'car', 'bike', 'car', 'car', 'car', 'truck', 'truck'],
                       'scores': [0.8328558802604675, 0.7974870204925537, 0.7834395170211792, 0.7779728174209595, 0.7673231363296509, 0.7419722676277161, 0.6387622356414795, 0.5442672967910767, 0.48913004994392395, 0.4428440034389496, 0.43411993980407715, 0.42757946252822876, 0.4275132119655609, 0.41344454884529114, 0.4089759588241577, 0.3834128975868225, 0.3770535886287689, 0.33668002486228943, 0.2713582217693329, 0.25302624702453613]}
    }

    nms_handler = NMSHandler()
    nms_handler.set_class_priorities({"car": 2, "bus": 1})
    nms_handler.set_model_priorities({"dino": 1, "yolo_world": 2})
    # nms_handler.set_iou_threshold(0.5)

    final_result = nms_handler.merge_results(formatted_result)
    print("Final Merged Result:", final_result)

    image_path = "/home/nehoray/PycharmProjects/UniversaLabeler/data/street/img.png"
    image = cv2.imread(image_path)
    nms_handler.save_result(image, final_result['bboxes'], final_result['labels'], final_result['scores'],
                            '/home/nehoray/PycharmProjects/UniversaLabeler/algorithms/detection_result.jpg')





