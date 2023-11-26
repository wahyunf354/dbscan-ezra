from flask import Flask, request
from flask_cors import CORS
from sklearn.cluster import DBSCAN

import pandas as pd

app = Flask(__name__)
app.config["DEBUG"] = True

CORS(app)

@app.route("/", methods=['GET', 'POST'])
def hello_world():
    if request.method == 'POST':
      data = request.json
      
      data_to_cluster = pd.DataFrame(data)

      lengthColumn = len(data_to_cluster.columns)
      # print(len(data_to_cluster.columns))
      dbscan = DBSCAN(eps=0.5, min_samples=2)
      dbscan.fit(data_to_cluster)
      # print(dbscan.labels_)

      data_to_cluster[lengthColumn] = dbscan.labels_       
      resultJSON = data_to_cluster.to_json(orient='records')
      return resultJSON, 200
    return "<p>Hello, World!</p>"


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
