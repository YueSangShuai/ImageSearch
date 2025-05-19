from imagesearch import ImageSearchAPI
from options import parse_args
from detect_person import PedestrianDetector
from PIL import Image


def additional_args(parser):
    parser.add_argument('--person_model', type=str, help='Path to ONNX model file', default="det_coco_s.onnx")
    parser.add_argument('--person_threshold', type=float, default=0.65, 
                       help='Detection confidence threshold for person (default: 0.5)')
    parser.add_argument('--enroll_path', type=str, default=[], nargs='+', help='Path to image files to enroll')
    parser.add_argument('--detection', help='Use detection mode', action='store_true')
    parser.add_argument('--check_exist', help='Check if the image is already registered', action='store_true')
    return parser


def enroll_main(args):
    image_search_api = ImageSearchAPI(args.model_path, args.device, args.qdrant_host, args.qdrant_port, args.collection_name)
    if args.detection:
        pedestrianDetector = PedestrianDetector(args.person_model, args.person_threshold)
        def detect_person(image_path):
            '''
                image_path: str
                return {""pil_image": image, "bbox": bbox}
            '''
            input_image = Image.open(image_path).convert("RGB")
            detections = pedestrianDetector.detect_image(image_pil=input_image, output_path=None)
            valid_detections = []
            for i, (x1, y1, x2, y2, score) in enumerate(detections):
                # crop image
                image = input_image.crop((x1, y1, x2, y2))
                valid_detections.append({"pil_image": image, "bbox": (x1, y1, x2, y2), "score": score})
            return valid_detections
    else:
        detect_person = None

    for image_path in args.enroll_path:
        image_search_api.register_new_images(image_path, args.category, check_exist=False, detector=detect_person)

    print("Enrollment completed.")
    print("\n")
    query_text = "a person wearing a red t-shirt"
    top_k = 5
    results, t = image_search_api.search(query_text, top_k, threshold=args.threshold, return_payload=True)
    if not results:
        print("No results found.")
        return
    print(results)
    print("Top {} results for query '{}', time: {}s:".format(top_k, query_text, t))
    for result in results:
        payload = result.payload
        path = payload['path']
        category = payload['category']
        bbox = payload['bbox'] if 'bbox' in payload else None
        score = result.score
        print(f"{path} ({category}, {bbox}): {score}")


if __name__ == '__main__':
    args = parse_args(additional_args)
    print(args)
    enroll_main(args)

