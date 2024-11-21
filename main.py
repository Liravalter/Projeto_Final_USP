import pandas as pd

class Modelo:
    def __init__(self):
        self.df = None

    def CarregarDataset('self, resultadoVale.csv'):
        self.df = pd.read_csv('resultadoVale.csv')
        print(self.df.head())  # Exibe as primeiras linhas do dataset
from sklearn.model_selection import train_test_split

    def TratamentoDeDados(self):
        # Verificar dados faltantes
        if self.df.isnull().sum().any():
            print("Dados faltantes encontrados!")
            # Tratar dados faltantes, por exemplo, com média ou mediana
            self.df = self.df.fillna(self.df.mean())

        # Divisão das variáveis independentes (X) e dependente (y)
        X = self.df.drop('Species', axis=1)  # Remove a coluna 'Species' para as features
        y = self.df['Species']  # A coluna 'Species' é o alvo
        
        # Convertendo a variável target para números (codificando as espécies)
        y = pd.Categorical(y).codes
        
        # Dividir em conjunto de treino e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Retornar os dados preparados
        return X_train, X_test, y_train, y_test
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

    def Treinamento(self, X_train, y_train):
        # Experimentando o modelo SVM
        modelo_svm = SVC(kernel='linear', random_state=42)
        modelo_svm.fit(X_train, y_train)
        
        # Experimentando a Regressão Logística (modelo de classificação)
        modelo_lr = LogisticRegression(max_iter=200, random_state=42)
        modelo_lr.fit(X_train, y_train)
        
        # Validando o modelo com cross-validation para comparar
        svm_cv_score = cross_val_score(modelo_svm, X_train, y_train, cv=5).mean()
        lr_cv_score = cross_val_score(modelo_lr, X_train, y_train, cv=5).mean()
        
        print(f'Acurácia com SVM: {svm_cv_score}')
        print(f'Acurácia com Regressão Logística: {lr_cv_score}')
        
        return modelo_svm, modelo_lr
from sklearn.metrics import accuracy_score

    def Teste(self, modelo_svm, modelo_lr, X_test, y_test):
        # Testar SVM
        y_pred_svm = modelo_svm.predict(X_test)
        acuracia_svm = accuracy_score(y_test, y_pred_svm)
        print(f'Acurácia SVM no conjunto de teste: {acuracia_svm}')
        
        # Testar Regressão Logística
        y_pred_lr = modelo_lr.predict(X_test)
        acuracia_lr = accuracy_score(y_test, y_pred_lr)
        print(f'Acurácia Regressão Logística no conjunto de teste: {acuracia_lr}')
class Modelo:
    def __init__(self):
        self.df = None

    def CarregarDataset(self, path):
        self.df = pd.read_csv(path)
        print(self.df.head())  # Exibe as primeiras linhas do dataset

    def TratamentoDeDados(self):
        if self.df.isnull().sum().any():
            print("Dados faltantes encontrados!")
            self.df = self.df.fillna(self.df.mean())

        X = self.df.drop('Species', axis=1)
        y = self.df['Species']
        y = pd.Categorical(y).codes
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        return X_train, X_test, y_train, y_test

    def Treinamento(self, X_train, y_train):
        modelo_svm = SVC(kernel='linear', random_state=42)
        modelo_svm.fit(X_train, y_train)
        
        modelo_lr = LogisticRegression(max_iter=200, random_state=42)
        modelo_lr.fit(X_train, y_train)
        
        svm_cv_score = cross_val_score(modelo_svm, X_train, y_train, cv=5).mean()
        lr_cv_score = cross_val_score(modelo_lr, X_train, y_train, cv=5).mean()
        
        print(f'Acurácia com SVM: {svm_cv_score}')
        print(f'Acurácia com Regressão Logística: {lr_cv_score}')
        
        return modelo_svm, modelo_lr

    def Teste(self, modelo_svm, modelo_lr, X_test, y_test):
        y_pred_svm = modelo_svm.predict(X_test)
        acuracia_svm = accuracy_score(y_test, y_pred_svm)
        print(f'Acurácia SVM no conjunto de teste: {acuracia_svm}')
        
        y_pred_lr = modelo_lr.predict(X_test)
        acuracia_lr = accuracy_score(y_test, y_pred_lr)
        print(f'Acurácia Regressão Logística no conjunto de teste: {acuracia_lr}')

    def Train(self, path):
        self.CarregarDataset(path)
        X_train, X_test, y_train, y_test = self.TratamentoDeDados()
        modelo_svm, modelo_lr = self.Treinamento(X_train, y_train)
        self.Teste(modelo_svm, modelo_lr, X_test, y_test)
