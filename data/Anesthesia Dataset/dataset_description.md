# Dataset Description

This dataset is based on a propofol anesthesia study, the experimental design of which is described in detail in [(Chennu *et al*. 2016)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004669). A full version of the dataset is open for download by the University of Cambridge [here](https://www.repository.cam.ac.uk/handle/1810/252736).

The following files contain approximately 7 minutes of spontaneous high-density EEG activity (EGI Hydrocel GSN), recorded from 3 participants during two different states: Wakefulness (Baseline) and Sedation (eyes-closed). Sedation was determined by a target plasma concentration of 1,200 ng/ml, controlled by a computerized syringe driver that achieved and maintained the required propofol infusion rate (Alaris TCI mode, using the Marsh model). A 10-minute equilibration period was allowed before the Sedation recordings, in order to attain a pharmacologically-steady state. Blood samples were also taken between the anesthetic state, in order to characterize the inter-individual variability and to confirm similarity to target concentrations.

EEG data have been filtered between 0.5 - 45 Hz, segmented into 10-second epochs, cleaned from artifacts, and re-referenced to the average of all channels. 


**EEGLAB format files**:

* Subject 1 - Wakefulness
  * *08-2010-anest 20100301 095.004.fdt*
  * *08-2010-anest 20100301 095.004.set*

* Subject 1 - Sedation
  * *08-2010-anest 20100301 095.015.fdt*
  * *08-2010-anest 20100301 095.015.set*

* Subject 2 - Wakefulness
  * *09-2010-anest 20100301 135.003.fdt*
  * *09-2010-anest 20100301 135.003.set*

* Subject 2 - Sedation
  * *09-2010-anest 20100301 135.021.fdt*
  * *09-2010-anest 20100301 135.021.set*

* Subject 3 - Wakefulness
  * *20-2010-anest 20100414 131.004.fdt* 
  * *20-2010-anest 20100414 131.004.set*

* Subject 3 - Sedation
  * *20-2010-anest 20100414 131.022.fdt* 
  * *20-2010-anest 20100414 131.022.set*
