{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "e80971e7",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import warnings\n",
    "import sys\n",
    "\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from urllib.parse import urlparse\n",
    "import mlflow\n",
    "import mlflow.sklearn\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.cluster import KMeans\n",
    "\n",
    "import logging"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "a86cd512",
   "metadata": {},
   "outputs": [],
   "source": [
    "logging.basicConfig(level=logging.WARN)\n",
    "logger = logging.getLogger(__name__)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "699ddcb9",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Колличество значений равных 1  438\n",
      "Колличество значений равных 0  439\n"
     ]
    }
   ],
   "source": [
    "if __name__ == \"__main__\":\n",
    "    warnings.filterwarnings(\"ignore\")\n",
    "    np.random.seed(40)\n",
    "\n",
    "    # Путь к файлу для анализа\n",
    "    csv_url = (r'C:\\Users\\mihai\\Desktop\\Vmeste project\\box_2.csv')\n",
    "    \n",
    "    try:\n",
    "        data = pd.read_csv(csv_url)\n",
    "        \n",
    "    except Exception as e:\n",
    "        logger.exception(\n",
    "            \"Unable to download training & test CSV, check your internet connection. Error: %s\", e\n",
    "        )\n",
    "    \n",
    "    # Создаем экземпляр класса StandardScaler для стандартного распределения\n",
    "    dataScaler = StandardScaler()\n",
    "    \n",
    "    # Применим метод fit_transform для получения набора данных dataScaled              \n",
    "    dataScaled = dataScaler.fit_transform(data.values) \n",
    "\n",
    "    with mlflow.start_run():\n",
    "        model = KMeans(n_clusters=2)\n",
    "        model.fit_predict(dataScaled)\n",
    "        \n",
    "        x = len(np.where(model.fit_predict(dataScaled)==1)[0])\n",
    "        y = len(np.where(model.fit_predict(dataScaled)==0)[0])\n",
    "\n",
    "        \n",
    "        print('Колличество значений равных 1 ', len(np.where(model.fit_predict(dataScaled)==1)[0]))\n",
    "        print('Колличество значений равных 0 ', len(np.where(model.fit_predict(dataScaled)==0)[0]))\n",
    "        \n",
    "        # Логируем параметры и метрики\n",
    "        mlflow.log_param('Колличество значений равных 1 ', x)\n",
    "        mlflow.log_param('Колличество значений равных 0 ', y)\n",
    "        mlflow.log_metric('Соотношение 0 к 1', (x+y)/x) \n",
    "           \n",
    "        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme\n",
    "\n",
    "        # Model registry does not work with file store\n",
    "        if tracking_url_type_store != \"file\":\n",
    "\n",
    "            mlflow.sklearn.log_model(model, \"model\", registered_model_name=\"KMeans\")\n",
    "        else:\n",
    "            mlflow.sklearn.log_model(model, \"model\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
