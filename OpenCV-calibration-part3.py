# openCV import
import cv2 as cv
import numpy as np
import os

# Keycode definitions
ESC_KEY = 27
Q_KEY = 113
U_KEY = 117  # Touche 'u' pour basculer entre image originale et corrigée
SPACE_KEY = 32  # Touche pour capturer une image

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

def performCalibration(pattern_size=(9,6)):
    """
    Effectue la calibration avec les images GoPro
    
    Args:
        pattern_size: Tuple (largeur, hauteur) du nombre de coins intérieurs
    
    Returns:
        tuple: (matrice_caméra, coefficients_distorsion) ou (None, None) si échec
    """
    print("\n=== Calibration des images GoPro ===")
    print(f"Taille de l'echiquier: {pattern_size[0]}x{pattern_size[1]} coins interieurs")
    print("Appuyez sur SPACE pour inclure une image dans la calibration")
    print("Appuyez sur ESC pour terminer et calculer la calibration")
    print("Appuyez sur une autre touche pour passer a l'image suivante")
    
    # Préparation des points 3D de l'échiquier
    objp = np.zeros((pattern_size[1] * pattern_size[0], 3), np.float32)
    objp[:,:2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    
    # Listes pour stocker les points
    objpoints = []  # Points 3D dans l'espace réel
    imgpoints = []  # Points 2D dans le plan image
    
    # Création de la fenêtre
    window = "Calibration GoPro"
    cv.namedWindow(window, cv.WINDOW_AUTOSIZE)
    
    capture_count = 0
    
    # Boucle sur toutes les images
    for i in range(1, 27):
        path = f"../calib_gopro/calib_gopro/GOPR84{i:02d}.JPG"
        image = cv.imread(path)
        
        if image is None:
            print(f"Erreur: Impossible de charger l'image {path}")
            continue
        
        # Détection des coins
        result, found, corners = findNDisplayChessBoardCorners(image.copy(), pattern_size)
        
        # Affichage des informations
        status_color = (0, 255, 0) if found else (0, 0, 255)
        status_text = "Echiquier DETECTE!" if found else "Echiquier NON detecte"
        
        cv.putText(result, f"Image {i}/26 - Captures: {capture_count}", 
                   (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv.putText(result, status_text, 
                   (10, 70), cv.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        cv.putText(result, "SPACE: capturer | ESC: terminer | Autre: suivant", 
                   (10, 110), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv.imshow(window, result)
        
        key = cv.waitKey(0)
        
        # SPACE : capturer l'image si l'échiquier est détecté
        if key == SPACE_KEY and found:
            objpoints.append(objp)
            imgpoints.append(corners)
            capture_count += 1
            print(f"Image {i} incluse pour la calibration! Total: {capture_count}")
        
        # ESC : terminer et calibrer
        elif key == ESC_KEY:
            break
    
    cv.destroyWindow(window)
    
    # Vérification du nombre d'images capturées
    if len(objpoints) < 3:
        print(f"\nErreur: Pas assez d'images capturees ({capture_count}). Minimum: 3")
        return None, None
    
    # Calibration
    print(f"\n=== Calcul de la calibration avec {capture_count} images ===")
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)
    
    if ret:
        print("\nCalibration reussie!")
        print(f"\nMatrice intrinseque de la camera:\n{mtx}")
        print(f"\nCoefficients de distorsion:\n{dist}")
        
        return mtx, dist
    else:
        print("Erreur lors de la calibration!")
        return None, None

def displayUndistortedImages(mtx, dist):
    """
    Affiche les images GoPro avec possibilité de les redresser
    
    Args:
        mtx: Matrice intrinsèque de la caméra
        dist: Coefficients de distorsion
    """
    # Création de la fenêtre d'affichage
    window = "Gopro - Original vs Corrigee"
    cv.namedWindow(window, cv.WINDOW_AUTOSIZE)
    
    print("\n=== Affichage des images GoPro ===")
    print("Appuyez sur U pour basculer entre image originale et corrigee")
    print("Appuyez sur ESC pour quitter")
    print("Appuyez sur une autre touche pour passer a l'image suivante")
    
    show_undistorted = False
    newcameramtx = None
    
    # Boucle sur les images
    i = 1
    while i <= 26:
        # Format du chemin avec zero-padding
        path = f"../calib_gopro/calib_gopro/GOPR84{i:02d}.JPG"
        image = cv.imread(path)
        
        # Vérification du chargement de l'image
        if image is None:
            print(f"Erreur: Impossible de charger l'image {path}")
            i += 1
            continue
        
        # Calcul de la nouvelle matrice de caméra (une seule fois)
        if newcameramtx is None:
            h, w = image.shape[:2]
            newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        
        # Préparation de l'affichage selon le mode
        if show_undistorted:
            # Image corrigée (undistorted)
            dst = cv.undistort(image, mtx, dist, None, newcameramtx)
            
            # Affichage côte à côte : originale à gauche, corrigée à droite
            comparison = np.hstack((image, dst))
            
            # Ajout de labels
            cv.putText(comparison, "ORIGINALE", (10, 50), 
                       cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            cv.putText(comparison, "CORRIGEE", (w + 10, 50), 
                       cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        else:
            # Image originale uniquement
            comparison = image.copy()
            cv.putText(comparison, "ORIGINALE", (10, 50), 
                       cv.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
        
        # Affichage du numéro d'image et des instructions
        cv.putText(comparison, f"Image {i}/26", (10, comparison.shape[0] - 60), 
                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv.putText(comparison, "U: basculer | ESC: quitter | Autre: suivant", 
                   (10, comparison.shape[0] - 20), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv.imshow(window, comparison)
        
        key = cv.waitKey(0)
        
        # Touche U : basculer entre image originale et comparaison
        if key == U_KEY:
            show_undistorted = not show_undistorted
            print(f"Mode: {'Comparaison originale/corrigee' if show_undistorted else 'Image originale'}")
            continue  # Réafficher la même image dans le nouveau mode
        
        # Touche ESC : quitter
        elif key == ESC_KEY:
            print("Fermeture de l'application")
            break
        
        # Autre touche : image suivante
        else:
            i += 1
    
    cv.destroyWindow(window)

def cleanup(window):
    """
    Ferme la fenêtre d'affichage
    
    Args:
        window: Nom de la fenêtre à fermer
    """
    cv.destroyWindow(window)

def main():
    """
    Fonction principale : calibration puis affichage des images redressées
    """
    print("=== Calibration de la camera GoPro ===")
    
    # Demander la taille de l'échiquier
    print("\nConfiguration de l'echiquier:")
    while True:
        try:
            width = int(input("Nombre de coins interieurs dans la largeur (ex: 9) : "))
            height = int(input("Nombre de coins interieurs dans la hauteur (ex: 6) : "))
            if width > 0 and height > 0:
                break
            print("Les valeurs doivent etre positives")
        except ValueError:
            print("Veuillez entrer des nombres entiers")
    
    pattern_size = (width, height)
    
    # Effectuer la calibration
    mtx, dist = performCalibration(pattern_size)
    
    if mtx is None or dist is None:
        print("Calibration echouee. Arret du programme.")
        return
    
    # Affichage des images avec possibilité de redressement
    displayUndistortedImages(mtx, dist)

# Point d'entrée du programme
if __name__ == "__main__":
    main()