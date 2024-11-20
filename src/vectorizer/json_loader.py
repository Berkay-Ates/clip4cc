import json
import os

from PIL import Image
import matplotlib.pyplot as plt

class JsonDataLoader():
    def __init__(self, json_path, image_folder,data_set):
        # JSON dosyasını yükleyin
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        self.image_folder = image_folder
        self.data_set = data_set
        self.image_folderA = self.image_folder+"A/"
        self.image_folderB = self.image_folder+"B/"

    def len(self):
        return len(self.data)

    def getitem(self, idx):
        # JSON verisini indeksleyin
        item = self.data[idx]
        
        # Resim yolunu oluşturun
        img_id = item["img_id"]
        if self.data_set == "spot_the_diff":
            image1_path = os.path.join(self.image_folder, f"{img_id}.png")
            image2_path = os.path.join(self.image_folder, f"{img_id}_2.png")
        else:
            image1_path = os.path.join(self.image_folderA, f"{img_id}.png")
            image2_path = os.path.join(self.image_folderB, f"{img_id}.png")
        
        # Metni alın (birden fazla cümle olabilir)
        sentences = item["sentences"]

        
        return [image1_path, image2_path, sentences]
    
    def draw_item(self,idx):
        # Resmin yolu
        image1_path = self.getitem(idx)[0]
        image2_path = self.getitem(idx)[1]
        text = self.getitem(idx)[2]

        # Resmi yükle
        image1 = Image.open(image1_path)
        image2 = Image.open(image2_path)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # 1 satır, 2 sütun
        plt.subplots_adjust(bottom=0.1)  # Alt tarafta daha fazla boşluk bırakmak için

        axes[0].imshow(image1)
        axes[0].axis('off')  

        axes[1].imshow(image2)
        axes[1].axis('off')  

        combined_caption = "\n".join(text)
        plt.figtext(0.5, 0.01, combined_caption, ha="center", fontsize=12)
        plt.show()


    def draw_item_detailed(self, img_idx, txt_idx, cos_sim, model_name):
        # Resmin yolu
        image1_path = self.getitem(img_idx)[0]
        image2_path = self.getitem(img_idx)[1]
        text = self.getitem(img_idx)[2][txt_idx]
        cos_sim_text = f"Cos-Sim: {cos_sim:.2f}"

        # Resimleri yükle
        image1 = Image.open(image1_path)
        image2 = Image.open(image2_path)

        # Grafik oluştur
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # 1 satır, 2 sütun
        plt.subplots_adjust(bottom=0.2)  # Alt metin için boşluk bırak

        # Başlık ekle
        fig.suptitle(f"Model: {model_name}", fontsize=16, fontweight='bold', y=0.92)

        # İlk resmi ekle
        axes[0].imshow(image1)
        axes[0].axis('off')  # Ekseni kaldır

        # İkinci resmi ekle
        axes[1].imshow(image2)
        axes[1].axis('off')  # Ekseni kaldır

        # Alt metni ekle
        combined_caption = f"Description: {text}\n{cos_sim_text}"
        plt.figtext(0.5, 0.05, combined_caption, ha="center", fontsize=12, wrap=True)

        # Grafik göster
        plt.show()