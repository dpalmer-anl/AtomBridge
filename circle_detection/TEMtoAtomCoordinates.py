#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 09:54:27 2025

@author: tawfiqurrakib
"""

import cv2
import numpy as np
from ase import Atoms
import ase.io
from sklearn.cluster import DBSCAN

def imageTOcoordinates(image, scale, atom_type, min_dist =20, param1 = 50, param2 = 20, min_radius = 5, max_radius = 20):
    """
    Finds atomic coordinates in a high-resolution TEM image.


    Args:
        image_path (str): The path to the input TEM image file.
        scale (float): The scale of the image in pixels per nanometer (px/nm).
        atom_type (str): The type of atom to be identified (e.g., 'Si', 'O').
                         This is used for labeling the output data.
        min_dist (int): Minimum distance between the centers of detected circles.
        param1 (int): Upper threshold for the internal Canny edge detector.
        param2 (int): Threshold for center detection.
        min_radius (int): Minimum circle radius.
        max_radius (int): Maximum circle radius.

    Returns:
        list: A list of dictionaries, where each dictionary represents an atom
              and contains its coordinates (in pixels and nanometers) and type.
              Returns an empty list if no image is found or no atoms are detected.
    """
    # Applying a Gaussian blur to reduce noise and improve circle detection
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    
    #identifying hyperparameters (probably from the paper)
    min_dist =20  #min_dist: Minimum distance between the centers of detected circles.
    param1 = 50 #Upper threshold for the internal Canny edge detector.
    param2 = 20 #Threshold for center detection.
    min_radius = 5 #Minimum circle radius in pixel
    max_radius = 20 #Maximum circle radius in pixel 
    scale = 56.69 #The scale of the image in pixels per nanometer (px/nm).
    atom_type = "Se"
    # Using Hough Circle Transform to find atoms (circles)
    circles = cv2.HoughCircles(
            blurred_image,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=min_dist,
            param1=param1,
            param2=param2,
            minRadius=min_radius,
            maxRadius=max_radius
        )
    
    atomic_coordinates = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Extract pixel coordinates
            x_px, y_px, radius_px = i[0], i[1], i[2]
    
            # Convert pixel coordinates to nanometers
            x_nm = x_px / scale
            y_nm = y_px / scale
    
            atomic_coordinates.append({
                'atom_type': atom_type,
                'x_pixel': int(x_px),
                'y_pixel': int(y_px),
                'x_nm': round(x_nm, 4),
                'y_nm': round(y_nm, 4),
                'radius_pixel': int(radius_px)
            })
    
    for atom in atomic_coordinates:
        x, y = atom['x_pixel'], atom['y_pixel']
        # Draw the outer circle
        cv2.circle(image, (x, y), atom['radius_pixel'], (0, 255, 0), 2)
        # Draw the center of the circle
        cv2.circle(image, (x, y), 2, (0, 0, 255), 3)
    
    cv2.imwrite('output.png', image)
    return atomic_coordinates

def save_coordinates_to_cif_wrong(coordinates, cif_filename='atomic_structure.cif', cell_z_nm=1.0):
    """
    Saves atomic coordinates to a CIF file using ASE.

    Args:
        coordinates (list): A list of atom coordinate dictionaries.
        cif_filename (str): The name of the output CIF file.
        cell_z_nm (float): The thickness of the 3D cell in nanometers, as TEM is a 2D projection.
    """

    positions = [[atom['x_nm'], atom['y_nm'], 5.0] for atom in coordinates]
    symbols = [atom['atom_type'] for atom in coordinates]

    # Create an ASE Atoms object
    atoms = Atoms(symbols=symbols, positions=positions)

    # Define a simulation cell. Since data is 2D, we create a cell
    # that fits the data with some padding and an arbitrary Z-height.
    # Convert to Angstrom for ASE cell definition (1 nm = 10 Å)
    min_x = min(p[0] for p in positions) - 1.0
    max_x = max(p[0] for p in positions) + 1.0
    min_y = min(p[1] for p in positions) - 1.0
    max_y = max(p[1] for p in positions) + 1.0
    
    cell_x = (max_x - min_x)
    cell_y = (max_y - min_y)
    cell_z = cell_z_nm * 10
    
    atoms.set_cell([cell_x, cell_y, cell_z])
    atoms.set_pbc(True) # Set periodic boundary conditions

    # Center atoms in the new cell
    #atoms.center(about=(0.0, 0.0, 0.0))
    
    # Write to CIF file
    ase.io.write(cif_filename, atoms, format='cif')
    print(f"Atomic structure saved to {cif_filename}")
    
def save_coordinates_to_cif(coordinates, cif_filename='atomic_structure.cif', cell_z_nm=1.0):
    """
    Finds the lattice, identifies the basis atoms, and saves a periodic
    unit cell to a CIF file using ASE.

    Args:
        coordinates (list): A list of atom coordinate dictionaries.
        cif_filename (str): The name of the output CIF file.
        cell_z_nm (float): The thickness of the 3D cell in nanometers, as TEM is a 2D projection.
    """
        
    if len(coordinates) < 3:
        print("Not enough atoms to determine lattice vectors.")
        return

    # --- 1. Find Lattice Vectors from Interatomic Distances ---
    positions = np.array([[c['x_nm'], c['y_nm']] for c in coordinates])
    
    # Calculate all interatomic vectors
    vectors = []
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            vectors.append(positions[j] - positions[i])
    vectors = np.array(vectors)

    # Find the most common nearest-neighbor distance by creating a histogram
    lengths = np.linalg.norm(vectors, axis=1)
    # Focus on the bottom 10% of distances to find the nearest neighbors
    hist, bin_edges = np.histogram(lengths, bins=100, range=(0, np.percentile(lengths, 10)))
    if np.sum(hist) == 0:
        print("Could not determine nearest-neighbor distance.")
        return
        
    nn_dist = bin_edges[np.argmax(hist)]
    
    # Filter for vectors that are close to the nearest-neighbor distance
    tolerance = 0.2 * nn_dist # 20% tolerance can be tuned
    candidate_vectors = vectors[np.abs(lengths - nn_dist) < tolerance]

    if len(candidate_vectors) < 5: # Need enough vectors for clustering
        print("Not enough candidate vectors found to determine lattice.")
        return

    # Normalize vectors to cluster them by direction
    normalized_vectors = candidate_vectors / np.linalg.norm(candidate_vectors, axis=1)[:, np.newaxis]
    
    # Use DBSCAN to find clusters of vectors pointing in similar directions
    clustering = DBSCAN(eps=0.15, min_samples=3).fit(normalized_vectors) # eps and min_samples may need tuning
    labels = clustering.labels_
    
    # Calculate the mean vector for each cluster
    cluster_centers = []
    for label in set(labels):
        if label == -1: continue # Ignore noise points from DBSCAN
        cluster_vectors = candidate_vectors[labels == label]
        cluster_centers.append(np.mean(cluster_vectors, axis=0))
        
    # We need at least two non-collinear vectors to define a 2D lattice
    if len(cluster_centers) < 2:
        print(f"Could not determine two distinct lattice vectors. Found {len(cluster_centers)}.")
        return
        
    # Pick the first vector as our first lattice vector 'a'
    a_vec = np.array(cluster_centers[0])
    b_vec = None
    
    # Find the first vector that is not parallel to 'a' to be 'b'
    for i in range(1, len(cluster_centers)):
        vec = np.array(cluster_centers[i])
        cosine_angle = np.dot(a_vec, vec) / (np.linalg.norm(a_vec) * np.linalg.norm(vec))
        # Check if the angle is not close to 0 or 180 degrees
        if abs(abs(cosine_angle) - 1.0) > 0.2: 
            b_vec = vec
            break

    if b_vec is None:
        print("Could not find a second non-collinear lattice vector.")
        return
        
    print(f"Found lattice vectors (nm): a={np.round(a_vec, 3)}, b={np.round(b_vec, 3)}")

    # --- 2. Define Unit Cell and Find Basis Atoms ---
    # ASE uses Angstroms (1 nm = 10 Å)
    cell = np.array([
        [a_vec[0]*10, a_vec[1]*10, 0],
        [b_vec[0]*10, b_vec[1]*10, 0],
        [0,           0,           cell_z_nm*10]
    ])
    
    # We need the inverse of the 2D part of the cell matrix to find fractional coordinates
    cell_2d_inv = np.linalg.inv(cell[:2, :2])
    
    basis_atoms = []
    basis_symbols = []
    
    # To avoid adding duplicate atoms, we store the fractional coordinates of atoms we've added
    added_frac_coords = []
    
    for atom in coordinates:
        pos_cartesian_2d = np.array([atom['x_nm']*10, atom['y_nm']*10])
        # Transform to fractional coordinates
        pos_frac_2d = np.dot(pos_cartesian_2d, cell_2d_inv)
        
        # Bring the fractional coordinates into the [0, 1) range
        pos_frac_2d -= np.floor(pos_frac_2d)
        
        # Check if a similar atom is already in our basis list to avoid duplicates
        is_duplicate = False
        for existing_frac in added_frac_coords:
            delta = pos_frac_2d - existing_frac
            delta -= np.round(delta) # account for periodicity
            if np.linalg.norm(delta) < 0.1: # tolerance for duplicate check
                is_duplicate = True
                break
        
        if not is_duplicate:
            added_frac_coords.append(pos_frac_2d)
            # We add the cartesian coords (relative to cell origin) to the basis list
            basis_atoms.append(list(np.dot(pos_frac_2d, cell[:2,:2])) + [cell[2,2]/2]) # center in z
            basis_symbols.append(atom['atom_type'])

    if not basis_atoms:
        print("Could not identify any basis atoms for the found lattice.")
        return
        
    # --- 3. Create ASE Atoms Object and Save to CIF ---
    atoms = Atoms(symbols=basis_symbols, positions=basis_atoms, cell=cell, pbc=[True, True, False])
    
    # ASE's CIF writer works better with scaled positions, so convert them
    atoms.set_scaled_positions(atoms.get_scaled_positions())

    ase.io.write(cif_filename, atoms, format='cif')
    print(f"Periodic unit cell with {len(basis_atoms)} basis atom(s) saved to {cif_filename}")






#load image
image_path = 'SeTe_TEM.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
scale = 56.69
atom_type = "Se"
atomic_coordinates = imageTOcoordinates(image, scale, atom_type, min_dist =20, param1 = 50, param2 = 20, min_radius = 5, max_radius = 20)
#save_coordinates_to_cif(atomic_coordinates, cif_filename='atomic_structure.cif', cell_z_nm=1.0)
positions = np.array([[c['x_nm'], c['y_nm']] for c in atomic_coordinates])
vectors = []
for i in range(len(positions)):
    for j in range(i + 1, len(positions)):
        vectors.append(positions[j] - positions[i])
vectors = np.array(vectors)
# Find the most common nearest-neighbor distance by creating a histogram
lengths = np.linalg.norm(vectors, axis=1)
# Focus on the bottom 10% of distances to find the nearest neighbors
hist, bin_edges = np.histogram(lengths, bins=10, range=(0, np.percentile(lengths, 10)))
if np.sum(hist) == 0:
    print("Could not determine nearest-neighbor distance.")
    
nn_dist = bin_edges[np.argmax(hist)]
# Filter for vectors that are close to the nearest-neighbor distance
tolerance = 0.2 * nn_dist # 20% tolerance can be tuned
candidate_vectors = vectors[np.abs(lengths - nn_dist) < tolerance]
if len(candidate_vectors) < 5: # Need enough vectors for clustering
    print("Not enough candidate vectors found to determine lattice.")

# Normalize vectors to cluster them by direction
normalized_vectors = candidate_vectors / np.linalg.norm(candidate_vectors, axis=1)[:, np.newaxis]
# Use DBSCAN to find clusters of vectors pointing in similar directions
clustering = DBSCAN(eps=0.15, min_samples=3).fit(normalized_vectors) # eps and min_samples may need tuning
labels = clustering.labels_

# Calculate the mean vector for each cluster
cluster_centers = []
for label in set(labels):
    if label == -1: continue # Ignore noise points from DBSCAN
    cluster_vectors = candidate_vectors[labels == label]
    cluster_centers.append(np.mean(cluster_vectors, axis=0))
    
    
    
    
