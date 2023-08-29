from fastapi import APIRouter, FastAPI, Security, File, UploadFile, Security,Form
from pydantic import BaseModel
from typing import Annotated, Union

import _pkl_flow

'''
import sys
sys.path.insert(0, 'C:/Users/rujirang.w/Downloads/FastAPI_Stackpython-main/FastAPI_Stackpython-main/routers')
from routers 
'''


router = APIRouter(
    prefix="/predict_images",
    tags=["demo predict structure"],
    responses={404: {"message": "Not found"}}
)

class_db = [
    {
        "select_diseases_name": "None",
        "class_boxes": "None",
        "confidence_value": "None",
        "match" : "Yes/No",
        "lat": "lat",
        "lng": "lng",
        "err_info": "None"
    } 
]

class Diseases(BaseModel):
    select_diseases_name: str
    class_boxes: str
    confidence: str
    match: str
    lat: str
    lng: str
    err_info: str

"""
@router.get("/")
async def get_qestion():
    print(len(Qboisduval_db))
    return Qboisduval_db


@router.get("/random")
async def get_qestion_random(api_key: str = Security(_secure.auth_api_key)):
    print(random.randint(0, len(Qboisduval_db)-1))
    return Qboisduval_db[random.randint(0, len(Qboisduval_db)-1)]
"""

@router.post("/predict/num_image")
async def predict_image( file: Annotated[bytes, File()], _diseases_name: Union[str, None] = None):
    
    if _diseases_name != None :
            slen = len(_diseases_name) 
    else:
            slen = 0

    if slen > 20 or slen == 0 :
       class_db[0]["err_info"] = "_disease_name is not_correct!"
       class_db[0]["match"] = "No"
       return class_db

    
    images = _pkl_flow.read_imagefile(file)
    '''
    prediction =_pkl_flow.image_predict(images)
    cls_b = str(prediction[0])
    conf_d = str(prediction[1])
    
    _ans = chk_data(_diseases_name,cls_b)
    class_db[0]["select_diseases_name"] = _diseases_name
    class_db[0]["class_boxes"] = cls_b
    class_db[0]["confidence_value"]  = conf_d
    class_db[0]["match"] = _ans
    class_db[0]["err_info"] = "None"
    '''
    
    #_pkl_flow.store_picture_to_db(cls_b,conf_d,file)


    return  class_db

def chk_data(_param_1, _param_2):
    _ans = "No"
    if str(_param_1) == str(_param_2) :
          _ans = "Yes"
    return _ans
