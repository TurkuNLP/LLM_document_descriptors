from dataclasses import dataclass

@dataclass
class Descriptor:
    descriptor_id: str
    descriptor: str
    explainer: str
    
    @property
    def text(self) -> str:
        return f"{self.descriptor}; {self.explainer}" if self.explainer else self.descriptor

@dataclass
class Document:
    doc_id: str
    text: str
    descriptors: list[Descriptor]
    
    @property
    def descriptor_explainers(self) -> list[str]:
        return [d.text for d in self.descriptors]