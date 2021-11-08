CLASSES_LIST = [
    'bike',
    'bus',
    'car',
    'motor',
    'person',
    'rider',
    'truck'
]

TRACK_CLASSES_LIST = [
    'car',
    'bus',
    'truck',
    'bike',
    'motor'
]

TRACK_CLASSES_LEN = [
    3, 
    8, 
    8,
    1,
    1
]

def get_cls_dict(category_num):
    """Get the class ID to name translation dictionary."""
    if category_num == len(CLASSES_LIST):
        return {i: n for i, n in enumerate(CLASSES_LIST)}
    else:
        return {i: 'CLS%d' % i for i in range(category_num)}
