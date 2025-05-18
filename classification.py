import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from classification.svm_classifier import predict_svm
import joblib
import os
import numpy as np

# Удаляем определение функции classify_image отсюда, чтобы избежать конфликта импорта
