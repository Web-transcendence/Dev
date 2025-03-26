void displayProgress(int current, int max) {
    int width = 20; // Number of total bar characters
    int progress = (current * width) / max; // Filled portion

    std::cout << "\r" << (current * 100) / max << "% [";
    for (int i = 0; i < width; i++) {
        std::cout << (i < progress ? '#' : '.');
    }
    std::cout << "]" << std::flush;
}