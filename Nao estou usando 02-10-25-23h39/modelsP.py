# models.py
from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class ResultadosOficiais(Base):
    __tablename__ = "resultados_oficiais"

    concurso = Column(Integer, primary_key=True)
    data = Column(String, nullable=False)  # Ex: '2024-03-01'
    n1 = Column(Integer)
    n2 = Column(Integer)
    n3 = Column(Integer)
    n4 = Column(Integer)
    n5 = Column(Integer)
    n6 = Column(Integer)
    n7 = Column(Integer)
    n8 = Column(Integer)
    n9 = Column(Integer)
    n10 = Column(Integer)
    n11 = Column(Integer)
    n12 = Column(Integer)
    n13 = Column(Integer)
    n14 = Column(Integer)
    n15 = Column(Integer)

    def to_dict(self):
        return {
            'concurso': self.concurso,
            'data': self.data,
            'numeros': [
                self.n1, self.n2, self.n3, self.n4, self.n5,
                self.n6, self.n7, self.n8, self.n9, self.n10,
                self.n11, self.n12, self.n13, self.n14, self.n15
            ]
        }