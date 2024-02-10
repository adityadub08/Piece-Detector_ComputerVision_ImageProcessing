from utils import Detect, LoadDataset

def main():
    input_folder = "./input/"
    output_folder = "./output/"

    dataset = LoadDataset(input_folder, output_folder)
    images_data = dataset.get_image_data()

    for image_path in images_data:
        image_name = image_path.split("/")[-1]
        print(f"started execution for {image_name}")
        detect = Detect(image_path)
        detect.detect_contours()
        detect.detect_corner()
        detect.show_edge_length()
        result_image, num_contours = detect.get_results()
        dataset.save_result(image_name, result_image)
        print(f"Saved Result for {image_name} and number of figures detected are {num_contours}")


if __name__ == "__main__":
    main()






