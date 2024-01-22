import os
from logging import getLogger


logger = getLogger(__name__)


def remove_sidebars(elements):
    from unstructured.documents.elements import ElementType
    from collections import defaultdict
    import re

    if not elements:
        return elements
    points_groups = defaultdict(list)
    min_x = 99999999
    max_x = 0
    e2index = {e.id: i for i, e in enumerate(elements)}
    for e in elements:
        x_l = int(e.metadata.coordinates.points[0][0])
        x_r = int(e.metadata.coordinates.points[2][0])
        points_groups[(x_l, x_r)].append(e)
        min_x = min(min_x, x_l)
        max_x = max(max_x, x_r)
    sidebars_elements = set()
    for (x_l, x_r), es in points_groups.items():
        first_id = e2index[es[0].id]
        last_id = e2index[es[-1].id]
        on_left = first_id == 0 and x_l == min_x
        on_right = (last_id == len(elements) - 2) and x_r == max_x
        loc_match = [on_left, on_right]
        total_text = "".join(map(str, es))
        condiction = [
            any(loc_match),
            len(es) >= 3,
            re.findall("^[A-Z\s\d,]+$", total_text),
        ]
        if not all(condiction):
            continue
        sidebars_elements.update(map(lambda x: x.id, es))
        if on_left:
            check_page_num_e = elements[last_id + 1]
        else:
            check_page_num_e = elements[-1]
        if (
            check_page_num_e.category == ElementType.UNCATEGORIZED_TEXT
            and check_page_num_e.text.strip().isalnum()
        ):
            sidebars_elements.add(check_page_num_e.id)

    elements = [e for e in elements if e.id not in sidebars_elements]
    return elements


def remove_annotation(elements):
    from collections import Counter
    from unstructured.documents.elements import ElementType

    page_num = max(e.metadata.page_number for e in elements)
    un_texts_counter = Counter(
        [e.text for e in elements if e.category == ElementType.UNCATEGORIZED_TEXT]
    )
    rm_text = set()
    for text, count in un_texts_counter.items():
        if count / page_num >= 0.5:
            rm_text.add(text)
    elements = [e for e in elements if e.text not in rm_text]
    return elements


def create_chunk_and_metadatas(page_elements, stride=3, window=10):
    page_elements = remove_sidebars(page_elements)
    for index, page_element in enumerate(page_elements):
        page_element.metadata.num = index
    datas = []
    for i in range(0, len(page_elements), stride):
        windown_elements = page_elements[i : i + window]
        chunk = "\n".join([e.text for e in windown_elements])
        source_elements = [e.to_dict() for e in windown_elements]
        datas.append(
            {
                "txt": chunk,
                "source_elements": source_elements,
            }
        )
    return datas


def get_chunks(elements):
    from collections import defaultdict
    from unstructured.documents.coordinates import RelativeCoordinateSystem

    elements = remove_annotation(elements)

    pages_elements = defaultdict(list)
    for element in elements:
        element.convert_coordinates_to_new_system(
            RelativeCoordinateSystem(), in_place=True
        )
        pages_elements[element.metadata.page_number].append(element)

    all_chunks_and_links = sum(
        [
            create_chunk_and_metadatas(page_elements)
            for _, page_elements in pages_elements.items()
        ],
        [],
    )
    return all_chunks_and_links


def rematch(texts, answer, threshold=0.0):
    texts_words = [set(t.split()) for t in texts]
    answer_words = set(answer.split())
    scores = [len(t & answer_words) / len(answer_words) for t in texts_words]
    max_score = max(scores)
    # make sure the threshold is not larger than the max score, otherwise, no results will be returned
    threshold = min(threshold, max_score)
    scores = [s if s >= threshold else 0 for s in scores]
    if sum(scores) == 0:
        return None, None
    max_index = scores.index(max_score)
    # Find the first non-zero score before the max score
    start = max_index
    for i in range(max_index - 1, -1, -1):
        if scores[i] < threshold:
            break
        start = i

    # Find the first non-zero score after the max score
    end = max_index + 1
    for i in range(max_index + 1, len(scores)):
        if scores[i] < threshold:
            break
        end = i + 1

    return start, end


def merge_metadatas(metadatas):
    if not metadatas:
        return {}
    metadata = metadatas[0]
    p1, p2, p3, p4 = metadata["coordinates"]["points"]
    corrdinate = [p1, p3]
    coordinates = [corrdinate]
    for metadata in metadatas[1:]:
        p1_, p2_, p3_, p4_ = metadata["coordinates"]["points"]
        if p2_[0] > p3[0]:
            corrdinate = [p1_, p3_]
            coordinates.append(corrdinate)
            p1, p2, p3, p4 = p1_, p2_, p3_, p4_
            continue
        p1 = (min(p1[0], p1_[0]), max(p1[1], p1_[1]))
        p2 = (min(p2[0], p2_[0]), max(p2[1], p2_[1]))
        p3 = (max(p3[0], p3_[0]), min(p3[1], p3_[1]))
        p4 = (max(p4[0], p4_[0]), min(p4[1], p4_[1]))
        corrdinate[0] = p1
        corrdinate[1] = p3

    page_number = metadata["page_number"]
    file_name = metadata["filename"]
    return {
        "file_name": file_name,
        "page_number": page_number,
        "coordinates": coordinates,
    }


def get_related_documents(contexts, match_text=None, filter_threshold=0.1):
    """
    Convert contexts to a dataframe
    """
    image_folder = os.environ.get("IMAGES_FOLDER", None)
    for source in contexts:
        chunk_data = source.outputs("elements", "chunk")
        source_elements = chunk_data["source_elements"]
        if match_text:
            start, end = rematch(
                [e["text"] for e in source_elements], match_text, filter_threshold
            )
            if start is None:
                continue
            source_elements = source_elements[start:end]
        metadata = merge_metadatas([e["metadata"] for e in source_elements])
        page_number = metadata["page_number"]
        file_name = metadata["file_name"]
        coordinates = metadata["coordinates"]
        file_path = os.path.join(image_folder, file_name, f"{page_number-1}.jpg")
        if os.path.exists(file_path):
            img = draw_rectangle_and_display(file_path, coordinates)
        else:
            img = None
        score = round(source["score"], 2)
        text = f"**file_name**: {file_name}\n\n**score**: {score}\n\n**text:**\n\n{chunk_data['txt']}"
        yield text, img

def get_related_merged_documents(contexts, match_text=None, filter_threshold=0.1):
    """
    Convert contexts to a dataframe
    Will merge the same page
    """
    image_folder = os.environ.get("IMAGES_FOLDER", None)

    page_elements, page2score = groupby_source_elements(contexts)
    for page_number, source_elements in page_elements.items():
        if match_text:
            start, end = rematch(
                [e["text"] for e in source_elements], match_text, filter_threshold
            )
            if start is None:
                continue
            source_elements = source_elements[start:end]
        text = "\n".join([e["text"] for e in source_elements])
        metadata = merge_metadatas([e["metadata"] for e in source_elements])
        file_name = metadata["file_name"]
        coordinates = metadata["coordinates"]
        file_path = os.path.join(image_folder, file_name, f"{page_number-1}.jpg")
        if os.path.exists(file_path):
            img = draw_rectangle_and_display(file_path, coordinates)
        else:
            img = None
        score = round(page2score[page_number], 2)
        text = (
            f"**file_name**: {file_name}\n\n**score**: {score}\n\n**text:**\n\n{text}"
        )
        yield text, img


def groupby_source_elements(contexts):
    """
    Group pages for all contexts
    """
    from collections import defaultdict

    # Save the max score for each page
    page2score = {}
    page_elements = defaultdict(list)
    for source in contexts:
        chunk_data = source.outputs("elements", "chunk")
        source_elements = chunk_data["source_elements"]
        for element in source_elements:
            page_number = element["metadata"]["page_number"]
            page_elements[page_number].append(element)

        page_number = chunk_data["source_elements"][0]["metadata"]["page_number"]
        score = source["score"]
        page2score[page_number] = max(page2score.get(page_number, 0), score)

    # Deduplicate elements in the page based on the num field
    for page_number, elements in page_elements.items():
        page_elements[page_number] = list(
            {e["metadata"]["num"]: e for e in elements}.values()
        )
        # Sort elements by num
        page_elements[page_number].sort(key=lambda e: e["metadata"]["num"])

    return page_elements, page2score


def draw_rectangle_and_display(image_path, relative_coordinates, expand=0.005):
    """
    Draw a rectangle on an image based on relative coordinates with the origin at the bottom-left
    and display it in Jupyter Notebook.

    :param image_path: Path to the original image.
    :param relative_coordinates: A list of (left, bottom, right, top) coordinates as a ratio of the image size.
    """
    from PIL import Image, ImageDraw

    with Image.open(image_path) as img:
        width, height = img.size
        draw = ImageDraw.Draw(img)

        # Convert relative coordinates to absolute pixel coordinates
        #
        for relative_coordinate in relative_coordinates:
            (left, top), (right, bottom) = relative_coordinate
            absolute_coordinates = (
                int(left * width),  # Left
                height - int(top * height),  # Top (inverted)
                int(right * width),  # Right
                height - int(bottom * height),  # Bottom (inverted)
            )

            if expand:
                absolute_coordinates = (
                    absolute_coordinates[0] - expand * width,
                    absolute_coordinates[1] - expand * height,
                    absolute_coordinates[2] + expand * width,
                    absolute_coordinates[3] + expand * height,
                )

            try:
                draw.rectangle(absolute_coordinates, outline="red", width=3)
            except Exception as e:
                logger.warn(
                    f"Failed to draw rectangle on image: {e}\nCoordinates: {absolute_coordinates}"
                )
        return img
