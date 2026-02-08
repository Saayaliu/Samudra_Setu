import os
import io
import tensorflow as tf
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd 

# Get the base directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'otolith_model.keras')

# Initialize FastAPI app
app = FastAPI()

# IMPORTANT: CORS settings to allow your React app to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained Keras model from the provided file
model = None
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"FATAL ERROR: Could not load model from {MODEL_PATH}: {e}")

# Define the image dimensions and class names based on model analysis
IMG_HEIGHT = 128
IMG_WIDTH = 128

# ######################################################################
# ###   CRITICAL CLASS NAME DEFINITIONS (PADDED LISTS)               ###
# ######################################################################
# These lists are padded to prevent IndexErrors (e.g., 180 names known, 478 expected).
# The error message now correctly indicates when the index is outside the known range.

# Output 1: Scientific Name (KNOWN: 180 / EXPECTED: 478)
SCIENTIFIC_NAMES_KNOWN = ["Abudefduf vaigiensis", "Acanthurus nigricans", "Acanthurus triostegus", "Aetobatus narinari", "Albula vulpes", "Aluterus monoceros", "Anchoa mitchilli", "Anguilla rostrata", "Antennarius striatus", "Apogon maculatus", "Archosargus probatocephalus", "Atherinella blackburni", "Atherinella beani", "Atherinomorus stipes", "Bagre marinus", "Balistes capriscus", "Barbulifer ceuthoecus", "Batrachoides marinus", "Bothus ocellatus", "Branchiostoma floridae", "Brotula barbata", "Brosmophycis marginata", "Bythites fuscus", "Bythites gerdae", "Bythites hildebrandi", "Calamus arctifrons", "Caranx crysos", "Caranx hippos", "Caranx ruber", "Carcharhinus leucas", "Carcharhinus limbatus", "Centropomus undecimalis", "Chaetodon capistratus", "Chaetodon ocellatus", "Chilomycterus schoepfii", "Chloroscombrus chrysurus", "Chriodorus atherinoides", "Chromis enchrysurus", "Clupea harengus", "Cochlespira kurodai", "Conger oceanicus", "Coryphaena hippurus", "Cyprinodon variegatus", "Dasyatis sabina", "Decapterus punctatus", "Diapterus auratus", "Diodon holocanthus", "Diplectrum formosum", "Echeneis naucrates", "Elacatinus oceanops", "Elops saurus", "Engraulis encrasicolus", "Epinephelus adscensionis", "Epinephelus itajara", "Eptatretus stoutii", "Etropus microstomus", "Etropus rimosus", "Eucinostomus argenteus", "Eucinostomus gula", "Eucinostomus havana", "Eucinostomus jonesii", "Eugerres plumieri", "Euthynnus alletteratus", "Fistularia tabacaria", "Fundulus heteroclitus", "Fundulus similis", "Galeocerdo cuvier", "Galeorhinus galeus", "Gambusia affinis", "Genidens genidens", "Ginglymostoma cirratum", "Gobiesox strumosus", "Gobiosoma evelynae", "Gobiosoma hemigymnum", "Gobiosoma macrognathos", "Gobiosoma schultzi", "Gymnothorax funebris", "Gymnothorax moringa", "Gymnothorax vicinus", "Haemulon aurolineatum", "Haemulon flavolineatum", "Haemulon plumierii", "Harengula jaguana", "Hippocampus erectus", "Hoplunnis diomedeana", 'Hoplunnis macrura', 'Hoplunnis pacifica', 'Hoplunnis punctata', 'Hoplunnis tenuis', 'Hypanus americana', 'Hyporhamphus unifasciatus', 'Istiophorus platypterus', 'Jenkinsia lamprotaenia', 'Lagodon rhomboides', 'Lepisosteus platyrhincus', 'Lutjanus apodus', 'Lutjanus griseus', 'Lutjanus jocu', 'Megalops atlanticus', 'Menticirrhus americanus', 'Menticirrhus saxatilis', 'Mugil curema', 'Mugil cephalus', 'Mycteroperca microlepis', 'Mycteroperca venenosa', 'Mycteroperca bonaci', 'Myrophis punctatus', 'Naucrates ductor', 'Negaprion brevirostris', 'Ocyurus chrysurus', 'Ophichthus gomesii', 'Ophidion holbrooki', 'Ophisurus macrorhynchos', 'Orthopristis chrysoptera', 'Pachyurus bonariensis', 'Palinurichthys perciformis', 'Parablennius marmoreus', 'Paralichthys dentatus', 'Pareques acuminatus', 'Peprilus alepidotus', 'Peprilus triacanthus', 'Petromyzon marinus', 'Phycis chesteri', 'Phycis cirrata', 'Phycis phycis', 'Pogonias cromis', 'Pomatomus saltatrix', 'Pomatoschistus microps', 'Pomatoschistus minutus', 'Pomatoschistus pictus', 'Priacanthus arenatus', 'Prionotus scitulus', 'Prionotus tribulus', 'Pseudupeneus maculatus', 'Pterois volitans', 'Rachycentron canadum', 'Raja binoculata', 'Raja clavata', 'Raja eglanteria', 'Raja erinacea', 'Raja inornata', 'Raja ocellata', 'Raja radula', 'Rhinecanthus aculeatus', 'Rhomboplites aurorubens', 'Sardinella aurita', 'Sardinella maderensis', 'Sardinops sagax', 'Saurida brasiliensis', 'Sciaenops ocellatus', 'Scomber scombrus', 'Scomberomorus maculatus', 'Serranus tabacarius', 'Sphyraena barracuda', 'Sphyrna lewini', 'Sphyrna mokarran', 'Sphyrna zygaena', 'Stephanolepis hispidus', 'Stramonita canaliculata', 'Symphurus civitatum', 'Symphurus plagusia', 'Syngnathus fuscus', 'Synodus foetens', 'Syrictes squamosus', 'Synodus intermedius', 'Tetrapturus albidus', 'Thalassoma bifasciatum', 'Thunnus alalunga', 'Thunnus albacares', 'Thunnus obesus', 'Thunnus thynnus', 'Trachinotus carolinus', 'Trachurus lathami', 'Trachurus trachurus', 'Triakis semifasciata', 'Tylosurus crocodilus', 'Umbrina coroides', 'Urophycis chuss', 'Urophycis regia', 'Vinciguerria attenuata']
SCIENTIFIC_NAMES = SCIENTIFIC_NAMES_KNOWN + [f'SCIENTIFIC_NAME_MISSING_INDEX_{i}' for i in range(len(SCIENTIFIC_NAMES_KNOWN), 478)]

# Output 2: Family (KNOWN: 85 / EXPECTED: 141)
FAMILY_NAMES_KNOWN = ["Acanthuridae", "Aetobatidae", "Albulidae", "Anguillidae", "Antennariidae", "Apogonidae", "Atherinopsidae", "Bagridae", "Balistidae", "Batrachoididae", "Bothidae", "Branchiostomatidae", "Brotulidae", "Bythitidae", "Carangidae", "Carcharhinidae", "Centropomidae", "Chaetodontidae", "Chilomycteridae", "Chloroscombridae", "Chromidae", "Clupeidae", "Congeridae", "Coryphaenidae", "Cyprinodontidae", "Dasyatidae", "Decapteridae", "Diapteridae", "Diodontidae", "Diplectridae", "Echeneidae", "Elacatinidae", "Elopsidae", "Engraulidae", "Epinephelidae", "Eptatretidae", "Etropidae", "Eucinostomidae", "Eugerresidae", "Euthynnidae", "Fistulariidae", "Fundulidae", "Galeocerdae", "Galeorhinidae", "Gambusidae", "Genidensidae", "Ginglymostomatidae", "Gobiesocidae", "Gobiosomatidae", "Gymnothoracidae", "Haemulonidae", "Harengulidae", "Hippocampidae", "Hoplunnidae", "Hypanidae", "Hyporhamphidae", "Istiophoridae", "Jenkinsidae", "Lagodonidae", "Lepisosteidae", "Lutjanidae", "Megalopidae", "Menticirrhidae", "Mugilidae", "Mycteropercidae", "Myrophidae", "Naucratesidae", "Negaprionidae", "Ocyuridae", "Ophichthidae", "Ophidionidae", "Ophisuridae", "Orthopristidae", "Pachyuridae", "Palinurichthyidae", "Parablennidae", "Paralichthyidae", "Parequesidae", "Peprilidae", "Petromyzonidae", "Phycidae", "Pogoniasidae", "Pomatomidae", "Pomatoschistidae", "Priacanthidae", "Prionotidae", "Pseudupeneidae", "Pteroidae", "Rachycentridae", "Rajidae", "Rhinecanthidae", "Rhomboplitesidae", "Sardinellidae", "Sardinopsidae", "Sauridae", "Sciaenopsidae", "Scomberidae", "Scomberomoridae", "Serranidae", "Sphyraenidae", "Sphyrnidae", "Stephanolepididae", "Stramonitidae", "Symphuridae", "Syngnathidae", "Synodusidae", "Tetrapturidae", "Thalassomatidae", "Thunnidae", "Trachinotusidae", "Trachuridae", "Triakisidae", "Tylosurusidae", "Umbrinidae", "Urophycidae", "Vinciguerriidae"]
FAMILY_NAMES = FAMILY_NAMES_KNOWN + [f'FAMILY_MISSING_INDEX_{i}' for i in range(len(FAMILY_NAMES_KNOWN), 141)]

# Output 3: Genus (KNOWN: 131 / EXPECTED: 318)
GENUS_NAMES_KNOWN = ["Abudefduf", "Acanthurus", "Aetobatus", "Albula", "Aluterus", "Anchoa", "Anguilla", "Antennarius", "Apogon", "Archosargus", "Atherinella", "Atherinomorus", "Bagre", "Balistes", "Barbulifer", "Batrachoides", "Bothus", "Branchiostoma", "Brotula", "Brosmophycis", "Bythites", "Calamus", "Caranx", "Carcharhinus", "Centropomus", "Chaetodon", "Chilomycterus", "Chloroscombrus", "Chriodorus", "Chromis", "Clupea", "Cochlespira", "Conger", "Coryphaena", "Cyprinodon", "Dasyatis", "Decapterus", "Diapterus", "Diodon", "Diplectrum", "Echeneis", "Elacatinus", "Elops", "Engraulis", "Epinephelus", "Eptatretus", "Etropus", "Eucinostomus", "Eugerres", "Euthynnus", "Fistularia", "Fundulus", "Galeocerdo", "Galeorhinus", "Gambusia", "Genidens", "Ginglymostoma", "Gobiesox", "Gobiosoma", "Gymnothorax", "Haemulon", "Harengula", "Hippocampus", "Hoplunnis", "Hypanus", "Hyporhamphus", "Istiophorus", "Jenkinsia", "Lagodon", "Lepisosteus", "Lutjanus", "Megalops", "Menticirrhus", "Mugil", "Mycteroperca", "Myrophis", "Naucrates", "Negaprion", "Ocyurus", "Ophichthus", "Ophidion", "Ophisurus", "Orthopristis", "Pachyurus", "Palinurichthys", "Parablennius", "Paralichthys", "Pareques", "Peprilus", "Petromyzon", "Phycis", "Pogonias", "Pomatomus", "Pomatoschistus", "Priacanthus", "Prionotus", "Pseudupeneus", "Pterois", "Rachycentron", "Raja", "Rhinecanthus", "Rhomboplites", "Sardinella", "Sardinops", "Saurida", "Sciaenops", "Scomber", "Scomberomorus", "Serranus", "Sphyraena", "Sphyrna", "Stephanolepis", "Stramonita", "Symphurus", "Syngnathus", "Synodus", "Syrictes", "Tetrapturus", "Thalassoma", "Thunnus", "Trachinotus", "Trachurus", "Triakis", "Tylosurus", "Umbrina", "Urophycis", "Vinciguerria"]
GENUS_NAMES = GENUS_NAMES_KNOWN + [f'GENUS_MISSING_INDEX_{i}' for i in range(len(GENUS_NAMES_KNOWN), 318)]

# Output 4: Specific Epithet (KNOWN: 157 / EXPECTED: 440)
EPITHET_NAMES_KNOWN = ["aculeatus", "adscensionis", "aeglefinus", "affinis", "alalunga", "albacares", "albidus", "alletteratus", "americana", "americanus", "apodus", "arenatus", "argenteus", "auratus", "aurita", "aurorubens", "aurolineatum", "attenuata", "barbata", "barracuda", "beani", "bifasciatum", "binoculata", "blackburni", "bonaci", "bonariensis", "brevirostris", "brasiliensis", "capistratus", "capriscus", "canadum", "canaliculata", "carolinus", "ceuthoecus", "cephalus", "chesteri", "chrysurus", "chuss", "cirrata", "cirratum", "civitatum", "clavata", "coroides", "crosos", "crocodilus", "cuvier", "dentatus", "ductor", "eglanteria", "encrasicolus", "enchrysurus", "erectus", "erinacea", "evelynae", "flavolineatum", "floridae", "foetens", "formosum", "funebris", "fuscus", "galeus", "gerdae", "gomesii", "griseus", "gula", "harengus", "havana", "hemigymnum", "heteroclitus", "hildebrandi", "hippos", "hippurus", "hispidus", "holocanthus", "holbrooki", "inornata", "intermedius", "itajara", "jaguana", "jocu", "jonesii", "kurodai", "lamprotaenia", "lathami", "leucas", "limbatus", "maculatus", "maderensis", "macrura", "macrorhynchos", "marinus", "marmoreus", "microlepis", "microps", 'microstomus', 'minutus', 'mitchilli', 'mokarran', 'monoceros', 'moringa', 'morhua', 'narinari', 'naucrates', 'niger', 'nigricans', 'occultus', 'ocellata', 'ocellatus', 'oceanicus', 'oceanops', 'obesus', 'pacifica', 'perciformis', 'phycis', 'pictus', 'platyrhincus', 'platypterus', 'plagusia', 'plumieri', 'plumierii', 'probatocephalus', 'punctata', 'punctatus', 'radula', 'regia', 'rimosus', 'rostrata', 'ruber', 'sabina', 'sagax', 'saltatrix', 'saurus', 'saxatilis', 'schoepfii', 'schultzi', 'semifasciata', 'similis', 'squamosus', 'stipes', 'strumosus', 'tabacaria', 'tenuis', 'thynnus', 'trachurus', 'triacanthus', 'tribulus', 'triostegus', 'undecimalis', 'unifasciatus', 'vaigiensis', 'variegatus', 'venenosa', 'vicinus', 'virens', 'volitans', 'vulpes', 'zygaena']
EPITHET_NAMES = EPITHET_NAMES_KNOWN + [f'EPITHET_MISSING_INDEX_{i}' for i in range(len(EPITHET_NAMES_KNOWN), 440)]

# Combine the lists for use in the prediction loop
ALL_CLASS_NAMES = [
    SCIENTIFIC_NAMES,
    FAMILY_NAMES,
    GENUS_NAMES,
    EPITHET_NAMES
]

# Labels for the frontend display
OUTPUT_LABELS = ["Scientific Name", "Family", "Genus", "Specific Epithet"]

# ######################################################################
# #################### END OF CRITICAL SECTION #########################
# ######################################################################


# Function to preprocess the uploaded image
def preprocess_image(image_bytes: bytes):
    try:
        # The model expects a color image (3 channels), so we convert to 'RGB'
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = image.resize((IMG_WIDTH, IMG_HEIGHT))
        image = np.array(image)
        image = image / 255.0  # Normalize pixel values [0, 1]
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

@app.get("/")
def read_root():
    return {"message": "Otolith Prediction API is running!"}

@app.post("/predict/")
async def predict_otolith(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Check server logs for model file path error.")

    image_bytes = await file.read()
    processed_img = preprocess_image(image_bytes)
    predictions = model.predict(processed_img)

    for i, pred_array in enumerate(predictions):
        if np.any(np.isnan(pred_array)) or np.any(np.isinf(pred_array)):
            raise HTTPException(
                status_code=422,
                detail=f"Model prediction for Output {i+1} resulted in invalid values (NaN/Inf). The model may be unstable or the input image is unusual."
            )

    results = []
    for i, pred_array in enumerate(predictions):
        score = tf.nn.softmax(pred_array[0])
        predicted_class_index = np.argmax(score)
        
        # Determine the size of the known data set for error reporting
        known_size = [180, 85, 131, 157][i]

        if predicted_class_index < known_size:
             # The index is within the range of known data, return the real name
             predicted_class_name = ALL_CLASS_NAMES[i][predicted_class_index]
        else:
             # The index falls into the padded, unknown range.
             predicted_class_name = f"ERROR: Index {predicted_class_index} is outside the known data range (Size: {known_size}). The true name is missing from the provided CSV data."
            
        confidence = float(np.max(score))

        results.append({
            "label": OUTPUT_LABELS[i],
            "prediction": predicted_class_name,
            "confidence": confidence
        })

    return JSONResponse(content=results)
