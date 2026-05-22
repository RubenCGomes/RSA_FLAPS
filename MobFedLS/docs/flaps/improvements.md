### Augmentation Techniques
1. Remixing (Stem Shuffling): Implemented a powerful technique that replaces stems in a mixture with the corresponding stems from different random tracks with a configurable probability (--augment-remix-prob). This breaks the natural correlation between instruments that always play together, forcing the model to learn the intrinsic characteristics of each instrument.
2. Phase Flipping: Added random signal phase inversion (--augment-phase-flip). This simple but effective augmentation doubles the effective variety of the training data without altering the sonic content.
3. Enhanced Normalization: Added a final normalization step in the dataset loader to ensure that the combined "remixed" mixtures do not clip before being processed by the model.

### Loss Function Improvements
1. SI-SDR Loss Integration: Added a differentiable implementation of SI-SDR (sisdr_loss) in the spectral domain. This directly optimizes for the primary metric used to evaluate source separation quality, leading to better perceptual results and faster convergence toward high-SDR models.
2. Refined Weight Balance: Updated the default loss weights to provide a more balanced training signal:
   * Mask Loss (L1): 35% (Reduced from 45%)
   * Magnitude Loss (Multi-scale): 35% (Reduced from 40%)
   * SI-SDR Loss (Spectral): 20% (New)
   * Consistency & KL Losses: 5% each (Reduced from 10%)

3. Tuned Defaults: These new weights are now the defaults in both the MLApp logic and the train_musdb.py script.