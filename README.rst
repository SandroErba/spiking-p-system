Spiking Neural P System Simulator
=================================

This repository contains a simulator for **Spiking Neural P Systems** (SNPS), developed in collaboration with the **University of Verona**.

The initial fork was taken from `spiking-p-system <https://github.com/a1sabau/spiking-p-system>`_.

----------------------

Features
--------

- Generate and load P systems  
- Run simulations and modify weights/rules  
- Compute energy costs  
- Analyze structural and performance aspects  
- Test different network architectures (image classification, edge detection, and experimental setups)

----------------------

Installation
------------

1. **Clone the repository**

   .. code-block:: bash

      git clone https://github.com/SandroErba/spiking-p-system.git
      cd spiking-p-system

2. **Set up a Python environment** (recommended: Python 3.9 or newer)

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate      # On macOS/Linux
      venv\Scripts\activate         # On Windows

3. **Install required dependencies**

   All dependencies are listed in the ``requirements.txt`` file.  
   You can install them with:

   .. code-block:: bash

      pip install -r requirements.txt 

----------------------

Configuration
-------------

Network configurations and parameters can be modified in the ``Config`` module.  
This allows you to explore and test different P System structures and behaviors.

Refer to the ``Technical_info.pdf`` document for a detailed explanation of:

- the internal structure and functioning of the simulator
- how networks are defined
- how to extend or modify them

----------------------

Running the Simulator
---------------------

To start the simulator, run:

.. code-block:: bash

   python main.py

From ``main.py``, you can test the main network modules:

- **``MedMnist``** – network for image classification tasks  
- **``EdgeDetection``** – network that generate edge-based images  
- **``OtherNetworks``** – small networks from literature and reference papers  

Each module represents a different use case of Spiking Neural P Systems.

----------------------

Documentation
-------------

- `Technical_info.pdf <Technical_info.pdf>`_ – detailed technical and theoretical background.  
- `Spiking Neural P Systems <https://www.semanticscholar.org/paper/Spiking-Neural-P-Systems-Ionescu-Paun/1db2b443a0fc71a3fae9a66c4ae16905a26baa17>`_  
  *Ionescu, Mihai, Gheorghe Păun, and Takashi Yokomori.*  
  *Fundamenta Informaticae*, 71.2–3 (2006): 279–308.


----------------------
