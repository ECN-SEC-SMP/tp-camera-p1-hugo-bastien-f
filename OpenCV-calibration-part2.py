# openCV import
import cv2 as cv
import numpy as np

# Keycode definitions
ESC_KEY = 27  # Touche pour capturer une image
Q_KEY = 113   # Touche pour quitter
U_KEY = 117   # Touche 'u' pour basculer entre image originale et undistorted

def openCAM(cam_id):
    """
    Ouvre la caméra avec l'identifiant spécifié
    
    Args:
        cam_id: Identifiant de la caméra (int)
    
    Returns:
        VideoCapture object ou None si échec
    """
    cam = cv.VideoCapture(cam_id)
    if not cam.isOpened():
        print(f"Erreur: Impossible d'ouvrir la caméra {cam_id}")
        return None
    return cam

def findNDisplayChessBoardCorners(image, pattern_size):
    """
    Détecte et affiche les coins de l'échiquier sur l'image
    
    Args:
        image: Image BGR à analyser
        pattern_size: Tuple (largeur, hauteur) du nombre de coins intérieurs
    
    Returns:
        tuple: (image avec coins dessinés, booléen de détection, coins raffinés)
    """
    # Conversion en niveaux de gris
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    # Recherche des coins de l'échiquier
    ret, corners = cv.findChessboardCorners(gray, pattern_size, 
                                           flags=cv.CALIB_CB_ADAPTIVE_THRESH)
    
    # Si l'échiquier est détecté, raffiner les coins
    if ret == True:
        # Critères de raffinement des coins
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # Raffinement sub-pixel des coins
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        # Dessiner les coins sur l'image
        cv.drawChessboardCorners(image, pattern_size, corners2, ret)
        return image, ret, corners2
    else:
        return image, ret, None

def getUserInputs():
    """
    Demande à l'utilisateur toutes les informations nécessaires pour la calibration
    
    Returns:
        tuple: (cam_id, pattern_width, pattern_height, num_images)
    """
    print("=== Configuration de la calibration ===")
    
    # Identifiant de la caméra
    while True:
        try:
            cam_id = int(input("Entrez l'identifiant de la caméra (généralement 0 ou 1) : "))
            break
        except ValueError:
            print("Erreur: Veuillez entrer un nombre entier")
    
    # Nombre de coins intérieurs dans la largeur
    while True:
        try:
            pattern_width = int(input("Nombre de coins intérieurs dans la largeur de l'échiquier : "))
            if pattern_width > 0:
                break
            print("Erreur: Le nombre doit être positif")
        except ValueError:
            print("Erreur: Veuillez entrer un nombre entier")
    
    # Nombre de coins intérieurs dans la hauteur
    while True:
        try:
            pattern_height = int(input("Nombre de coins intérieurs dans la hauteur de l'échiquier : "))
            if pattern_height > 0:
                break
            print("Erreur: Le nombre doit être positif")
        except ValueError:
            print("Erreur: Veuillez entrer un nombre entier")
    
    # Nombre d'images pour la calibration
    while True:
        try:
            num_images = int(input("Nombre d'images à capturer pour la calibration (minimum 3, recommandé 10-20) : "))
            if num_images >= 3:
                break
            print("Erreur: Il faut au moins 3 images pour calibrer")
        except ValueError:
            print("Erreur: Veuillez entrer un nombre entier")
    
    return cam_id, pattern_width, pattern_height, num_images

def calibrationLoop(cap, pattern_size, num_images, window):
    """
    Boucle de calibration : capture des images avec l'échiquier
    
    Args:
        cap: VideoCapture object
        pattern_size: Tuple (largeur, hauteur) du nombre de coins intérieurs
        num_images: Nombre d'images à capturer
        window: Nom de la fenêtre d'affichage
    
    Returns:
        tuple: (objpoints, imgpoints, dernière_image) ou (None, None, None) si échec
    """
    # Préparation des points 3D de l'échiquier (coordonnées réelles)
    objp = np.zeros((pattern_size[1] * pattern_size[0], 3), np.float32)
    objp[:,:2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    
    # Listes pour stocker les points de toutes les images
    objpoints = []  # Points 3D dans l'espace réel
    imgpoints = []  # Points 2D dans le plan image
    
    capture_count = 0
    last_image = None
    
    print(f"\n=== Boucle de calibration ===")
    print(f"Objectif: Capturer {num_images} images")
    print("Appuyez sur ESC pour capturer une image")
    print("Appuyez sur Q pour quitter")
    
    while capture_count < num_images:
        # Capture d'une frame
        ret, image = cap.read()
        
        if not ret:
            print("Erreur: Impossible de lire la frame")
            return None, None, None
        
        # Copie de l'image pour l'affichage
        display_image = image.copy()
        
        # Détection des coins de l'échiquier
        result, found, corners = findNDisplayChessBoardCorners(display_image, pattern_size)
        
        # Affichage des informations sur l'image
        status_color = (0, 255, 0) if found else (0, 0, 255)
        status_text = "Echiquier detecte!" if found else "Echiquier NON detecte"
        
        cv.putText(result, f"Images capturees: {capture_count}/{num_images}", 
                   (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv.putText(result, status_text, 
                   (10, 70), cv.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        cv.putText(result, "ESC: capturer | Q: quitter", 
                   (10, 110), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv.imshow(window, result)
        
        key = cv.waitKey(1)
        
        # Touche ESC : capturer l'image si l'échiquier est détecté
        if key == ESC_KEY:
            if found:
                objpoints.append(objp)
                imgpoints.append(corners)
                capture_count += 1
                last_image = image.copy()
                print(f"Image {capture_count}/{num_images} capturee avec succes!")
            else:
                print("Impossible de capturer: Echiquier non detecte!")
        
        # Touche Q : quitter
        elif key == Q_KEY:
            print("Calibration annulee par l'utilisateur")
            return None, None, None
    
    print(f"\n{num_images} images capturees avec succes!")
    return objpoints, imgpoints, last_image

def performCalibration(objpoints, imgpoints, image_shape):
    """
    Calcule les paramètres de calibration de la caméra
    
    Args:
        objpoints: Liste des points 3D de l'échiquier
        imgpoints: Liste des points 2D détectés
        image_shape: Forme de l'image (hauteur, largeur)
    
    Returns:
        tuple: (matrice_caméra, coefficients_distorsion) ou (None, None) si échec
    """
    print("\n=== Calibration en cours ===")
    print("Calcul des parametres...")
    
    # Calibration de la caméra
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, image_shape[::-1], None, None)
    
    if ret:
        print("\nCalibration reussie!")
        print(f"\nMatrice intrinseque de la camera:\n{mtx}")
        print(f"\nCoefficients de distorsion:\n{dist}")
        
        return mtx, dist
    else:
        print("Erreur lors de la calibration!")
        return None, None

def displayLoop(cap, mtx, dist, window):
    """
    Boucle d'affichage : affiche l'image originale ou corrigée selon le choix utilisateur
    
    Args:
        cap: VideoCapture object
        mtx: Matrice intrinsèque de la caméra
        dist: Coefficients de distorsion
        window: Nom de la fenêtre d'affichage
    """
    print("\n=== Boucle d'affichage ===")
    print("Appuyez sur U pour basculer entre image originale et corrigee")
    print("Appuyez sur Q pour quitter")
    
    show_undistorted = False
    
    # Pré-calcul de la nouvelle matrice pour l'optimisation
    first_frame = True
    newcameramtx = None
    
    while True:
        # Capture d'une frame
        ret, image = cap.read()
        
        if not ret:
            print("Erreur: Impossible de lire la frame")
            break
        
        # Calcul de la nouvelle matrice à la première frame
        if first_frame:
            h, w = image.shape[:2]
            newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
            first_frame = False
        
        # Choix d'affichage selon le mode
        if show_undistorted:
            # Correction de la distorsion
            display_image = cv.undistort(image, mtx, dist, None, newcameramtx)
            mode_text = "Mode: IMAGE CORRIGEE"
            color = (0, 255, 0)
        else:
            # Image originale
            display_image = image.copy()
            mode_text = "Mode: IMAGE ORIGINALE"
            color = (255, 0, 0)
        
        # Affichage des informations
        cv.putText(display_image, mode_text, 
                   (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv.putText(display_image, "U: basculer | Q: quitter", 
                   (10, 70), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv.imshow(window, display_image)
        
        key = cv.waitKey(1)
        
        # Touche U : basculer entre image originale et corrigée
        if key == U_KEY:
            show_undistorted = not show_undistorted
            print(f"Basculement vers: {'Image corrigee' if show_undistorted else 'Image originale'}")
        
        # Touche Q : quitter
        elif key == Q_KEY:
            print("Fermeture de l'application")
            break

def cleanup(window, cap):
    """
    Nettoyage : ferme les fenêtres et libère la caméra
    
    Args:
        window: Nom de la fenêtre à fermer
        cap: VideoCapture object à libérer
    """
    cv.destroyWindow(window)
    if cap is not None:
        cap.release()

def main():
    """
    Fonction principale : orchestre tout le processus de calibration
    """
    # 1. Récupération des paramètres utilisateur
    cam_id, pattern_width, pattern_height, num_images = getUserInputs()
    pattern_size = (pattern_width, pattern_height)
    
    # 2. Ouverture de la caméra
    cap = openCAM(cam_id)
    if cap is None:
        return
    
    # 3. Création de la fenêtre d'affichage
    window = "Calibration Camera"
    cv.namedWindow(window, cv.WINDOW_AUTOSIZE)
    
    # 4. Boucle de calibration
    objpoints, imgpoints, last_image = calibrationLoop(cap, pattern_size, num_images, window)
    
    # Vérification que la calibration n'a pas été annulée
    if objpoints is None or imgpoints is None:
        cleanup(window, cap)
        return
    
    # 5. Calcul des paramètres de calibration
    gray = cv.cvtColor(last_image, cv.COLOR_BGR2GRAY)
    mtx, dist = performCalibration(objpoints, imgpoints, gray.shape)
    
    if mtx is None or dist is None:
        cleanup(window, cap)
        return
    
    # 6. Boucle d'affichage
    displayLoop(cap, mtx, dist, window)
    
    # 7. Nettoyage
    cleanup(window, cap)

# Point d'entrée du programme
if __name__ == "__main__":
    main()