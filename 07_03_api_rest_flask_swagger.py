# revisa http://localhost:5000/openapi/
from http import HTTPStatus
from pydantic import BaseModel, Field

from flask_openapi3 import Info, Tag
from flask_openapi3 import OpenAPI
from typing import Annotated, Dict, List, Literal, Tuple
info = Info(title="iris API", version="1.0.0", description="API documentation")
# Basic Authentication Sample
basic = {
  "type": "http",
  "scheme": "basic"
}
# JWT Bearer Sample
jwt = {
  "type": "http",
  "scheme": "bearer",
  "bearerFormat": "JWT"
}
# API Key Sample
api_key = {
  "type": "apiKey",
  "name": "api_key",
  "in": "header"
}
# Implicit OAuth2 Sample
oauth2 = {
  "type": "oauth2",
  "flows": {
    "implicit": {
      "authorizationUrl": "https://example.com/api/oauth/dialog",
      "scopes": {
        "write:pets": "modify pets in your account",
        "read:pets": "read your pets"
      }
    }
  }
}
security_schemes = {"jwt": jwt, "api_key": api_key, "oauth2": oauth2, "basic": basic}
security = [
    {"jwt": []},
    {"oauth2": ["write:pets", "read:pets"]}
]
class NotFoundResponse(BaseModel):
    code: int = Field(-1, description="Status Code")
    message: str = Field("Resource not found!", description="Exception Information")

app = OpenAPI(
    __name__,
    info=info,
    security_schemes=security_schemes,
    responses={404: NotFoundResponse})

iris_tag = Tag(name="iris", description="Some API")


class APIQuery(BaseModel):
    features: List[float] = Field([],
                                  description='some iris data measures',
                                  # examples=["0.2", "0.2", "0.2", "0.2"]
                                  )

class APIResponse(BaseModel):
    code: int = Field(-1, description="Status Code")
    message: str = Field(..., description="Exception Information")
    prediction: int = Field(..., description="Prediction")
    target: str = Field(..., description="Target")
@app.post(
    "/predict",
    summary="predict iris",
    tags=[iris_tag],
    responses={200: APIResponse},
    security=security
    #request=APIQuery(features=[0.1, 0.2, 0.3, 0.4]),
)
def post_predict(body: APIQuery) -> APIResponse:
    """
    to get all books
    """
    print(body)
    # input_data = APIQuery.parse_obj(request.json)
    # print(input_data)
    return {
        "code": 0,
        "message": "ok",
        "data": [
             {"features": body}
        ]
    }, HTTPStatus.OK


if __name__ == "__main__":
    app.run(debug=True)
