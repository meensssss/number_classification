import uvicorn
from fastapi import FastAPI, File, UploadFile, Security,Form
from starlette.responses import RedirectResponse
from typing import Union
from typing import Annotated
from io import BytesIO
import sys
#sys.path.insert(0, 'C:/Users/rujirang.w/Desktop/disease_struc - pkl - only')
#import _pkl_flow, security 

#sys.path.insert(0, 'C:/Users/rujirang.w/Desktop/disease_struc - pkl - only/routers')
from routers import image_predict

app = FastAPI(title='Demo_Classifier Picture API')

@app.get("/", include_in_schema=False)
async def index():
    return RedirectResponse(url="/docs")

'''
@app.post("/predict/num_image")
async def predict_image( file: Annotated[bytes, File()], api_key: str = Security(security.auth_api_key)):
 
    images = _pkl_flow.read_imagefile(file)
    prediction =_pkl_flow.image_predict(images)

    #_pkl_flow.store_picture_to_db(cls_b,conf_d,file)
    
    return str(prediction[0])
'''

'''
@app.post("/api/model_index")
async def index_predict(no_index):
    print("number: ", no_index)
    pindex = _pkl_flow.x_predict(no_index)
    print("answer: ", pindex)
    return int(pindex)
'''

'''
@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}
'''

'''
@app.post("/files/")
async def create_file(
    file: Annotated[bytes, File()],
    fileb: Annotated[UploadFile, File()],
    token: Annotated[str, Form()],
):
    _pkl_flow.store_picture_to_db('_ex',file)
    return {
        "file_size": len(file),
        "token": token,
        "fileb_content_type": fileb.content_type,
    }
'''

app.include_router(image_predict.router)

'''
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, debug=True)
'''
