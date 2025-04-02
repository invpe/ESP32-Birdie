/*
  -------------------------------------------
  BIRD SOUND RECOGNITION SYSTEM (ESP32 + M5Stack)
  -------------------------------------------

  📌 WHAT THIS PROGRAM DOES:
  - Listens to audio via onboard microphone.
  - When it detects a loud sound – it records a short sample.
  - Computes FFT and normalizes the data (ignores background noise).
  - Sample is saved to internal FS.
  - If the model already knows the sound – it classifies and labels it instantly.
  - If not – the label is left empty.
  - Samples are accessible via WiFi through a built-in HTML page.
  - You can assign labels (e.g. "Tit", "Sparrow") manually.
  - Once enough labeled samples are gathered – you can train the model (mini-batch perceptron).
  - The trained model is automatically saved to `/model`.

  ⚙️ HOW TO USE:
  - Open the ESP32 IP address in your browser.
  - Review the list of recorded samples.
  - Assign labels (class names) and click "Save".
  - Click "Train" to train the AI model.

  📚 BEST PRACTICES FOR TRAINING:
  1. **Record in a quiet environment** – avoid background noise.
  2. **Each class should have at least 8–10 samples** (more is better), but keep ESP's memory in mind!
  3. **Tune the learning rate carefully** – recommended between 0.001 and 0.05 with 200–500 epochs.
  4. **You can train the model multiple times** – ideally overnight (e.g., at 03:00 AM).
     Re-training won’t hurt if you have new data.
  5. **You don’t need to keep old samples**, but:
     - If you want to **retrain from scratch**, you’ll need them.
     - If you just want to **continue training**, new labeled ones are enough.
  6. **Make a backup of the `/model` file** for safety (via `/file?name=model` for example).

  🚀 TIP: It’s a good idea to keep a few well-labeled samples
          from each class as "anchors" for future training sessions.

  👨‍💻 Author: invpe https://github.com/invpe/ESP32-Birdie
  📅 Version: 2025-04
*/



#include <map>
#include <vector>
#include <algorithm>
#include <M5Stack.h>
#include <driver/i2s.h>
#include <math.h>
#include <ArduinoOTA.h>
#include <utility>   

#include <ESPmDNS.h>
#include <WebServer.h>
#include <Adafruit_NeoPixel.h>
#include "FS.h"
#include "SPIFFS.h"

#define WIFI_AP "AAAAAAAAAAAAAAAAAAAAAAA"
#define WIFI_PASS "BBBBBBBBBBBBBBBBBBBBBB"

#define CONFIG_I2S_BCK_PIN -1
#define CONFIG_I2S_LRCK_PIN 5
#define CONFIG_I2S_DATA_PIN -1
#define CONFIG_I2S_DATA_IN_PIN 19

#define SPEAK_I2S_NUMBER I2S_NUM_0
#define MODE_MIC 0

#define SPIFFS_MIN_SIZE 10000  // Minimum spiffs size
#define DATA_SIZE 1024         // Rozmiar bloku odczytu (w bajtach)
#define THRESHOLD 3000         // Próg wykrywania dźwięku (dla pojedynczego bloku)
#define RECORD_BLOCKS 8        // Liczba bloków do rejestracji segmentu
#define RECORD_BUFFER_SIZE (DATA_SIZE * RECORD_BLOCKS)

const char* MODEL_FILE = "/model";
const char* LABEL_POSTFIX = "_label";
const char* FFT_POSTFIX = "_fft";

#define INPUT_SIZE 256  // Rozmiar wektora FFT
#define MAX_CLASSES 10  // Maksymalna liczba klas

WebServer http(80);
File fFileUpload;
uint64_t totalBytes = 0;
uint64_t usedBytes = 0;
uint64_t freeBytes = 0;
const char* ntpServer = "pool.ntp.org";
const long gmtOffset_sec = 3600;
const int daylightOffset_sec = 3600;

uint8_t audioBuffer[DATA_SIZE];            // Bufor do ciągłego nasłuchiwania
uint8_t recordBuffer[RECORD_BUFFER_SIZE];  // Bufor do zapisu segmentu audio
time_t timeSinceEpoch = 0;
float fLastTrainingError = 0.0;
bool recording = false;
int iAudioLevel = 0;
bool bAutoTrain = true;
// ---------------------
// Model perceptronu
// ---------------------

double weights[MAX_CLASSES][INPUT_SIZE / 2];
double biases[MAX_CLASSES];       // Bias dla każdej klasy
String classLabels[MAX_CLASSES];  // Etykiety klas

int numClasses = 0;

/*--------------*/
#define ADA_PIN 27
#define ADA_NUMPIXELS 1
Adafruit_NeoPixel pixels = Adafruit_NeoPixel(ADA_NUMPIXELS, ADA_PIN, NEO_GRB + NEO_KHZ800);


/*--------------*/
String epochToLocalTime(time_t epoch) {
  struct tm timeinfo;
  localtime_r(&epoch, &timeinfo);
  char buffer[20];  // Dla formatu "DD/MM/YYYY hh:mm:ss" wystarczy 20 znaków
  strftime(buffer, sizeof(buffer), "%d/%m/%Y %H:%M:%S", &timeinfo);
  return String(buffer);
}

void initModel() {
  numClasses = 0;
  for (int i = 0; i < MAX_CLASSES; i++) {
    biases[i] = 0;
    for (int j = 0; j < INPUT_SIZE / 2; j++) {
      // Losowa wartość z zakresu -0.01 do 0.01
      weights[i][j] = ((double)random(-100, 101)) / 1000.0;  // teraz zakres -0.1 do 0.1
    }
    classLabels[i] = "";
  }
}

bool InitI2SMicrophone() {
  esp_err_t err = ESP_OK;
  i2s_driver_uninstall(SPEAK_I2S_NUMBER);
  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX | I2S_MODE_PDM),
    .sample_rate = 16000,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
    .channel_format = I2S_CHANNEL_FMT_ALL_RIGHT,
#if ESP_IDF_VERSION > ESP_IDF_VERSION_VAL(4, 1, 0)
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
#else
    .communication_format = I2S_COMM_FORMAT_I2S,
#endif
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 6,
    .dma_buf_len = 60,
  };

  err += i2s_driver_install(SPEAK_I2S_NUMBER, &i2s_config, 0, NULL);

  i2s_pin_config_t pin_config;
#if (ESP_IDF_VERSION > ESP_IDF_VERSION_VAL(4, 3, 0))
  pin_config.mck_io_num = I2S_PIN_NO_CHANGE;
#endif
  pin_config.bck_io_num = CONFIG_I2S_BCK_PIN;
  pin_config.ws_io_num = CONFIG_I2S_LRCK_PIN;
  pin_config.data_out_num = CONFIG_I2S_DATA_PIN;
  pin_config.data_in_num = CONFIG_I2S_DATA_IN_PIN;

  err += i2s_set_pin(SPEAK_I2S_NUMBER, &pin_config);
  err += i2s_set_clk(SPEAK_I2S_NUMBER, 16000, I2S_BITS_PER_SAMPLE_16BIT, I2S_CHANNEL_MONO);

  return (err == ESP_OK);
}

// Funkcja nakładająca okno Hamming na dane
void applyHammingWindow(double* data, int n) {
  for (int i = 0; i < n; i++) {
    data[i] *= (0.54 - 0.46 * cos(2 * PI * i / (n - 1)));
  }
}

// Prosta implementacja FFT metodą Cooley-Tukey (in-place)
void fft(double* real, double* imag, int n) {
  int j = 0;
  for (int i = 0; i < n; i++) {
    if (i < j) {
      double temp = real[i];
      real[i] = real[j];
      real[j] = temp;
      temp = imag[i];
      imag[i] = imag[j];
      imag[j] = temp;
    }
    int m = n >> 1;
    while (m >= 1 && j >= m) {
      j -= m;
      m >>= 1;
    }
    j += m;
  }

  for (int s = 1; (1 << s) <= n; s++) {
    int m = 1 << s;
    double angle = -2 * PI / m;
    double w_m_real = cos(angle);
    double w_m_imag = sin(angle);
    for (int k = 0; k < n; k += m) {
      double w_real = 1.0;
      double w_imag = 0.0;
      for (int j = 0; j < m / 2; j++) {
        int t = k + j;
        int u = t + m / 2;
        double t_real = w_real * real[u] - w_imag * imag[u];
        double t_imag = w_real * imag[u] + w_imag * real[u];
        real[u] = real[t] - t_real;
        imag[u] = imag[t] - t_imag;
        real[t] += t_real;
        imag[t] += t_imag;

        double temp = w_real;
        w_real = temp * w_m_real - w_imag * w_m_imag;
        w_imag = temp * w_m_imag + w_imag * w_m_real;
      }
    }
  }
}
int processAudio(double* fftMagnitudes, int fftSize) {

  // Przepisz próbki audio z recordBuffer (int16_t) na dane typu double:
  int16_t* audioSamples = (int16_t*)recordBuffer;

  // Stworzenie tablic roboczych:
  double vReal[fftSize];
  double vImag[fftSize];

  // Wypełnienie danymi audio i zerowanie części urojonej:
  for (int i = 0; i < fftSize; i++) {
    vReal[i] = (double)audioSamples[i];
    vImag[i] = 0.0;
  }

  // Nakładamy okno Hamminga:
  applyHammingWindow(vReal, fftSize);

  // Obliczamy FFT:
  fft(vReal, vImag, fftSize);

  // Obliczamy magnitudy FFT (pełne spektrum):
  double maxMagnitude = 0.0;
  for (int i = 0; i < fftSize / 2; i++) {
    fftMagnitudes[i] = sqrt(vReal[i] * vReal[i] + vImag[i] * vImag[i]);
    if (fftMagnitudes[i] > maxMagnitude)
      maxMagnitude = fftMagnitudes[i];
  }

  // Normalizacja magnitud FFT do zakresu 0-1:
  if (maxMagnitude > 0) {
    for (int i = 0; i < fftSize / 2; i++)
      fftMagnitudes[i] /= maxMagnitude;
  }

  // Usunięcie DC offsetu
  fftMagnitudes[0] = 0.0;

  // Zwracamy maksymalną magnitudę (można użyć do wykrycia aktywności audio):
  return (int)maxMagnitude;
}


// Funkcja rejestrująca segment audio o zadanym rozmiarze (blokowym)
bool recordSegment(uint8_t* buffer, size_t bufferSize) {
  size_t offset = 0;
  while (offset < bufferSize) {
    size_t bytesRead = 0;
    i2s_read(SPEAK_I2S_NUMBER, (char*)(buffer + offset), DATA_SIZE, &bytesRead, (100 / portTICK_PERIOD_MS));
    if (bytesRead > 0) {
      offset += bytesRead;
    }
  }
  return true;
}

// Funkcja zapisująca WAV - budujemy nagłówek i zapisujemy dane audio
bool saveWavFile(const char* filename, uint8_t* data, size_t dataSize) {
  File file = SPIFFS.open(filename, FILE_WRITE);
  if (!file) {
    Serial.print("[ERROR] Cant open WAV for write: ");
    Serial.println(filename);
    return false;
  }
  uint32_t chunkSize = 36 + dataSize;
  uint16_t numChannels = 1;
  uint32_t sampleRate = 16000;
  uint16_t bitsPerSample = 16;
  uint32_t byteRate = sampleRate * numChannels * bitsPerSample / 8;
  uint16_t blockAlign = numChannels * bitsPerSample / 8;
  uint32_t subchunk2Size = dataSize;

  uint8_t header[44];
  memcpy(header, "RIFF", 4);
  header[4] = (chunkSize & 0xFF);
  header[5] = (chunkSize >> 8) & 0xFF;
  header[6] = (chunkSize >> 16) & 0xFF;
  header[7] = (chunkSize >> 24) & 0xFF;
  memcpy(header + 8, "WAVE", 4);
  memcpy(header + 12, "fmt ", 4);
  header[16] = 16;
  header[17] = 0;
  header[18] = 0;
  header[19] = 0;  // Subchunk1Size
  header[20] = 1;
  header[21] = 0;  // Audio format PCM
  header[22] = numChannels & 0xFF;
  header[23] = (numChannels >> 8) & 0xFF;
  header[24] = sampleRate & 0xFF;
  header[25] = (sampleRate >> 8) & 0xFF;
  header[26] = (sampleRate >> 16) & 0xFF;
  header[27] = (sampleRate >> 24) & 0xFF;
  header[28] = byteRate & 0xFF;
  header[29] = (byteRate >> 8) & 0xFF;
  header[30] = (byteRate >> 16) & 0xFF;
  header[31] = (byteRate >> 24) & 0xFF;
  header[32] = blockAlign & 0xFF;
  header[33] = (blockAlign >> 8) & 0xFF;
  header[34] = bitsPerSample & 0xFF;
  header[35] = (bitsPerSample >> 8) & 0xFF;
  memcpy(header + 36, "data", 4);
  header[40] = subchunk2Size & 0xFF;
  header[41] = (subchunk2Size >> 8) & 0xFF;
  header[42] = (subchunk2Size >> 16) & 0xFF;
  header[43] = (subchunk2Size >> 24) & 0xFF;

  file.write(header, 44);
  file.write(data, dataSize);
  file.close();

  return true;
}
String predictLabel(double* input, int inputSize, const double& dThreshold, float& fPewnosc) {
  if (numClasses == 0) {
    fPewnosc = 0;
    return "";
  }

  double scores[MAX_CLASSES];
  // Obliczamy score dla każdej klasy
  for (int i = 0; i < numClasses; i++) {
    double s = biases[i];
    for (int j = 0; j < inputSize; j++) {
      s += weights[i][j] * input[j];
    }
    scores[i] = s;
  }

  // Stabilizacja softmax – odejmujemy maksymalny score
  double maxScore = scores[0];
  for (int i = 1; i < numClasses; i++) {
    if (scores[i] > maxScore) {
      maxScore = scores[i];
    }
  }

  double expScores[MAX_CLASSES];
  double sumExp = 0;
  for (int i = 0; i < numClasses; i++) {
    expScores[i] = exp(scores[i] - maxScore);
    if (isnan(expScores[i]) || isinf(expScores[i])) {
      Serial.println("[ERROR] Wrong expScores for: " + classLabels[i]);
    }
    sumExp += expScores[i];
  }

  double bestProb = 0;
  int bestIndex = -1;
  for (int i = 0; i < numClasses; i++) {
    double prob = expScores[i] / sumExp;
    if (prob > bestProb) {
      bestProb = prob;
      bestIndex = i;
    }
  }

  // Ustawiamy pewność jako procent (np. 0.9 -> 90%)
  fPewnosc = bestProb * 100.0f;

  double thresholdProb = dThreshold;  // Próg pewności softmax (np. 0.8)
  if (bestProb < thresholdProb) {
    //Serial.print("[INFO] Softmax too low: ");
    //Serial.println(bestProb, 4);
    return "";
  }

  /*
  // Obliczamy cosine similarity między input a wagami dla bestIndex
  double dot = 0, normInput = 0, normWeight = 0;
  for (int j = 0; j < inputSize; j++) {
    dot += input[j] * weights[bestIndex][j];
    normInput += input[j] * input[j];
    normWeight += weights[bestIndex][j] * weights[bestIndex][j];
  }
  normInput = sqrt(normInput);
  normWeight = sqrt(normWeight);
  double cosine = 0;
  if (normInput > 0 && normWeight > 0) {
    cosine = dot / (normInput * normWeight);
  }

  double thresholdCosine = dThreshold;  // Próg dla cosine similarity, możesz ustawić osobno, jeśli chcesz
  Serial.print("[INFO] Cosine Similarity for ");
  Serial.print(classLabels[bestIndex]);
  Serial.print(" = ");
  Serial.println(cosine, 4);

  if (cosine < thresholdCosine) {
    //Serial.println("Cosine similarity poniżej progu, predykcja odrzucona.");
    return "";
  }
*/
  return classLabels[bestIndex];
}




bool saveModel(const float& fError) {
  File file = SPIFFS.open(MODEL_FILE, FILE_WRITE);
  if (!file) {
    Serial.println("[ERROR] Cant save model!");
    return false;
  }

  file.printf("%1.2f\n", fError);

  file.printf("numClasses:%d\n", numClasses);
  for (int i = 0; i < numClasses; i++) {
    file.printf("label:%s\n", classLabels[i].c_str());
    file.printf("bias:%f\n", biases[i]);
    // Zapisujemy wagi oddzielone przecinkami
    for (int j = 0; j < INPUT_SIZE / 2; j++) {
      file.printf("%f", weights[i][j]);
      if (j < (INPUT_SIZE / 2) - 1) file.print(",");
    }
    file.println();
  }
  file.close();
  return true;
}

bool loadModel(float& fError) {
  if (!SPIFFS.exists(MODEL_FILE)) {
    Serial.println("[INFO] Model file does not exist.");
    return false;
  }
  File file = SPIFFS.open(MODEL_FILE, "r");
  if (!file) {
    Serial.println("[ERROR] Can't open model file");
    return false;
  }

  String error = file.readStringUntil('\n');
  fError = error.toDouble();

  String line = file.readStringUntil('\n');
  if (line.startsWith("numClasses:")) {
    numClasses = line.substring(11).toInt();
  }
  for (int i = 0; i < numClasses; i++) {
    line = file.readStringUntil('\n');  // label
    if (line.startsWith("label:")) {
      classLabels[i] = line.substring(6);
      classLabels[i].trim();
    }
    line = file.readStringUntil('\n');  // bias
    if (line.startsWith("bias:")) {
      biases[i] = line.substring(5).toDouble();
    }
    line = file.readStringUntil('\n');  // weights
    int index = 0;
    int start = 0;
    while (index < INPUT_SIZE / 2) {
      int comma = line.indexOf(',', start);
      String token;
      if (comma == -1) {
        token = line.substring(start);
        weights[i][index] = token.toDouble();
        break;
      } else {
        token = line.substring(start, comma);
        weights[i][index] = token.toDouble();
        index++;
        start = comma + 1;
      }
    }
  }
  file.close();
  Serial.println("[INFO] Model loaded, last training error rate " + error);
  return true;
}

//FFT generuje N/2 unikalnych magnitud, zapisujemy N/2.
// Dane sa juz znormalizowane
bool loadFFTData(const String& fftFilename, double* input, int n) {
  if (!SPIFFS.exists(fftFilename)) {
    Serial.println("[ERROR] FFT File not found " + fftFilename);
    return false;
  }

  File fftFile = SPIFFS.open(fftFilename, "r");
  if (!fftFile) {
    Serial.println("[ERROR] Cant open FFT file " + fftFilename);
    return false;
  }

  int index = 0;
  while (fftFile.available() && index < n) {
    String line = fftFile.readStringUntil('\n');
    line.trim();
    if (line.length() > 0) {
      input[index++] = line.toDouble();
    }
  }
  fftFile.close();
  return (index == n);
}

//FFT generuje N/2 unikalnych magnitud, zapisujemy N/2.
// Dane sa juz znormalizowane
bool saveFFTData(const char* filename, double* fftData, int n) {
  File file = SPIFFS.open(filename, FILE_WRITE);
  if (!file) {
    Serial.print("[ERROR] Cant save FFT file ");
    Serial.println(filename);
    return false;
  }
  for (int i = 0; i < n; i++) {
    file.println(fftData[i], 4);  // zapisujemy z dokładnością do 4 miejsc po przecinku, każda wartość w nowej linii
  }
  file.close();
  return true;
}
String GetLabelForSample(const String& rSample) {
  if (SPIFFS.exists("/" + rSample + "_label")) {
    File fLabel = SPIFFS.open("/" + rSample + "_label");

    String label = fLabel.readString();
    label.trim();
    fLabel.close();
    return label;
  }

  return "";
}

// Funkcja trenuje model z wybranymi parametrami learn rate i epochami.
void Training(const float& fLR, const int& iEpochs) {
  pixels.setPixelColor(0, 255, 0, 0);
  pixels.show();


  File root = SPIFFS.open("/");
  File file = root.openNextFile();
  std::vector<std::pair<String, String>> batch;

  // Szukamy plików etykiet i tworzymy pary (nazwa próbki, etykieta)
  while (file) {
    String filename = String(file.name());

    if (filename.endsWith(LABEL_POSTFIX)) {
      // Nazwa próbki: usuwamy "/" i sufiks
      String sampleName = filename;
      sampleName.replace(LABEL_POSTFIX, "");

      // Odczytujemy etykietę
      File fLabel = SPIFFS.open("/" + filename, "r");
      String label = fLabel.readString();
      label.trim();
      fLabel.close();

      batch.push_back(std::make_pair(sampleName, label));
    }
    file = root.openNextFile();
  }

  Serial.println("[INFO] Training started for " + String(batch.size()) + " samples, learning rate : " + String(fLR) + " Epochs " + String(iEpochs));

  // Fisher-Yates shuffle:
  for (size_t i = batch.size() - 1; i > 0; i--) {
    size_t j = random(0, i + 1);  // Losowy indeks z zakresu [0, i]
    std::swap(batch[i], batch[j]);
  }

  double err = 0;
  for (int e = 0; e < iEpochs; e++) {
    // tasowanie próbek co epokę pomaga w lepszej generalizacji
    for (size_t i = batch.size() - 1; i > 0; i--) {
      size_t j = random(0, i + 1);
      std::swap(batch[i], batch[j]);
    }

    err = PerformMiniBatch(batch, fLR);
    Serial.println("[INFO] Epoch: " + String(e + 1) + "/" + String(iEpochs) + " Training error: " + String(err));
    fLastTrainingError = err;
    saveModel(err);
  }
  pixels.setPixelColor(0, 0, 0, 0);
  pixels.show();
}
double PerformMiniBatch(const std::vector<std::pair<String, String>>& batch, const double& dLearningRate) {

  // Dynamiczna alokacja gradientów na stercie
  double* gradB = (double*)calloc(MAX_CLASSES, sizeof(double));
  double** gradW = (double**)malloc(MAX_CLASSES * sizeof(double*));
  for (int i = 0; i < MAX_CLASSES; i++) {
    gradW[i] = (double*)calloc(INPUT_SIZE / 2, sizeof(double));
  }


  int validCount = 0;
  double totalErrorBatch = 0;  // Sumaryczny błąd dla całego batcha


  // Przetwarzamy każdą próbkę w batchu
  for (auto& sample : batch) {
    String sampleName = sample.first;                     // np. "123456" – nazwa próbki
    String label = sample.second;                         // etykieta, np. "Wróbel"
    String fftFilename = "/" + sampleName + FFT_POSTFIX;  // np. "/123456_fft"
    //Serial.println(sampleName + "/" + label);

    // Alokujemy pamięć na dane FFT
    double* input = (double*)malloc((INPUT_SIZE / 2) * sizeof(double));
    if (input == NULL) continue;
    if (!loadFFTData(fftFilename, input, INPUT_SIZE / 2)) {
      Serial.println("[ERROR] Cant load FFT for " + fftFilename);
      free(input);
      continue;
    }

    // Znajdź indeks klasy docelowej (dodajemy nową, jeśli jeszcze nie istnieje)
    int targetIndex = -1;
    for (int i = 0; i < numClasses; i++) {
      if (classLabels[i] == label) {
        targetIndex = i;
        break;
      }
    }
    if (targetIndex == -1) {
      if (numClasses < MAX_CLASSES) {
        targetIndex = numClasses;
        classLabels[numClasses] = label;

        // Inicjalizacja wag i biasu dla nowej klasy:
        biases[numClasses] = 0;
        for (int j = 0; j < INPUT_SIZE / 2; j++) {
          weights[numClasses][j] = ((double)random(-100, 101)) / 10000.0;
        }
        Serial.println("[INFO] Creating new class " + label);
        numClasses++;
      } else {
        free(input);
        continue;
      }
    }

    // Forward pass dla tej próbki
    double scores[MAX_CLASSES];
    double expScores[MAX_CLASSES];
    double sumExp = 0;
    for (int i = 0; i < numClasses; i++) {
      double s = biases[i];
      for (int j = 0; j < INPUT_SIZE / 2; j++) {
        s += weights[i][j] * input[j];
      }
      scores[i] = s;
    }
    double maxScore = scores[0];
    for (int i = 1; i < numClasses; i++) {
      if (scores[i] > maxScore)
        maxScore = scores[i];
    }
    for (int i = 0; i < numClasses; i++) {
      expScores[i] = exp(scores[i] - maxScore);
      sumExp += expScores[i];
    }

    // Obliczamy gradient dla każdej klasy oraz sumaryczny błąd dla próbki
    double sampleError = 0;
    for (int i = 0; i < numClasses; i++) {
      double target = (i == targetIndex) ? 1.0 : 0.0;
      double error = expScores[i] / sumExp - target;
      sampleError += fabs(error);
      gradB[i] += error;
      for (int j = 0; j < INPUT_SIZE / 2; j++) {
        gradW[i][j] += error * input[j];
      }
    }
    totalErrorBatch += sampleError;

    validCount++;
    free(input);
  }

  // Uśredniamy gradienty i aktualizujemy model
  if (validCount > 0) {


    for (int i = 0; i < numClasses; i++) {
      double avgGradB = gradB[i] / validCount;
      biases[i] -= dLearningRate * avgGradB;
      for (int j = 0; j < INPUT_SIZE / 2; j++) {
        double avgGradW = gradW[i][j] / validCount;
        weights[i][j] -= dLearningRate * avgGradW;
      }
    }
  }

  // Zwolnienie pamięci dynamicznej
  free(gradB);
  for (int i = 0; i < MAX_CLASSES; i++) {
    free(gradW[i]);
  }
  free(gradW);

  // Zwracamy średni błąd dla batcha (czyli totalErrorBatch / validCount)
  if (validCount > 0)
    return totalErrorBatch / validCount;
  else
    return -1;
}

void setup() {
  M5.begin();
  M5.Power.begin();

  pixels.begin();
  pixels.setPixelColor(0, 255, 255, 255);
  pixels.setBrightness(255);
  pixels.show();

  Serial.begin(115200);

  Serial.println("SPIIFS...");
  while (!SPIFFS.begin()) {
    SPIFFS.format();
    Serial.println("[ERROR] Failed to mount file system");
    delay(1000);
  }

  Serial.println("[INFO] Microphone init...");
  if (!InitI2SMicrophone()) {
    Serial.println("[ERROR] I Can't init microphone.");
  }

  // WIFI
  WiFi.hostname("ESP_BIRDIE");
  WiFi.begin(WIFI_AP, WIFI_PASS);
  uint8_t iRetries = 0;
  while (WiFi.status() != WL_CONNECTED) {
    Serial.println(".");
    delay(1000);
    iRetries += 1;
    if (iRetries >= 10) {
      Serial.println("No wifi");
      ESP.restart();
    }
  }

  http.on("/", []() {
    String html = "<html><head><meta charset='UTF-8'><title>Small Perceptron</title>"
                  "<style>"
                  "body { background-color: #000; color: #ddd; font-family: Arial, sans-serif; margin: 0; padding: 0; }"
                  "header, footer { background-color: #111; padding: 10px 20px; width: 100%; position: fixed; left: 0; z-index: 100; }"
                  "header { top: 0; }"
                  "footer { bottom: 0; }"
                  "main { padding: 120px 20px 80px 20px; max-width: 1000px; margin: auto; }"
                  "h1 { color: #66ccff; margin-top: 0; }"
                  "a { color: #66ccff; text-decoration: none; margin: 0 5px; }"
                  "a:hover { color: #ffcc00; }"
                  "table { width: 100%; border-collapse: collapse; margin-top: 20px; }"
                  "th, td { padding: 10px; border: 1px solid #333; text-align: center; }"
                  "input[type='text'] { width: 100px; }"
                  "input[type='submit'] { background-color: #66ccff; color: #000; border: none; padding: 5px 10px; }"
                  "input[type='submit']:hover { background-color: #ffcc00; }"
                  "</style></head><body>";

    // === HEADER ===
    html += "<header>";
    html += "<div style='display: flex; align-items: center; gap: 20px; flex-wrap: wrap;'>";

    // RAM (ikona 🧠)
    html += "<div style='color:#aaa;'>🧠 Heap: " + String(ESP.getFreeHeap()) + "b</div>";

    // SPIFFS (ikona 💾)
    html += "<div style='color:#aaa;'>💾 Spiffs: " + String(freeBytes) + "b / " + String(SPIFFS_MIN_SIZE) + "b</div>";

    // Sound level (ikona 🔊)
    html += "<div style='display: flex; align-items: center; gap: 8px;'>";
    html += "<span style='color:#aaa;'>🔊</span>";
    html += "<input type='range' id='levelSlider' min='0' max='" + String(THRESHOLD) + "' value='0' disabled style='width:150px;'>";
    html += "<span id='levelDisplay' style='font-size:14px; color:#ffcc00;'>...</span>";
    html += "</div>";


    // Linki + statystyki
    html += "<div style='display: flex; align-items: center; gap: 10px; font-size: 14px; flex-wrap: wrap;'>";


    html += "<a href='/deleteallsamples' style='color:#ff6666;'>🗑️ Remove Samples</a>";
    html += "<span style='color:#666;'>|</span>";

    html += "<a href='/delmodel' style='color:#ff6666;'>❌ Remove model</a>";
    html += "<span style='color:#666;'>|</span>";

    html += "<a href='/file?name=model' style='color:#66ccff;'>📁 Get Model</a>";
    html += "<span style='color:#666;'>|</span>";


    html += "<span style='color:#aaa;'>🎯 Model error: " + String(fLastTrainingError) + " |</span>";
    html += "<span style='color:#aaa;'>✨ Model classes: " + String(numClasses) + " |</span>";
    html += "<span style='color:#aaa;'>⌚ AutoTrain: <a href=/autotrain>" + String(bAutoTrain ? "Yes" : "No") + "</a> |</span>";

    html += "</div>";  // zamknięcie głównego diva


    html += "<script>";
    html += "function updateLevel() {";
    html += "  fetch('/level').then(r => r.text()).then(t => {";
    html += "    document.getElementById('levelSlider').value = t;";
    html += "    document.getElementById('levelDisplay').innerText = 'Level: ' + t;";
    html += "  });";
    html += "}";
    html += "setInterval(updateLevel, 500);";
    html += "updateLevel();";
    html += "</script>";
    html += "</header>";

    // === MAIN ===
    html += "<main>";
    html += "<h2>Samples recorded</h2>";
    html += "This is a list of samples recorded by the device.<BR>";
    html += "You can label them for training, or remove if not interesting.<BR>";
    html += "<table><tr><th>Time</th><th>File</th><th>Audio</th><th>Label</th><th>Action</th></tr>";

    std::vector<String> sampleFiles;

    // Zlap liste plikow ktore sie kwalifikuja
    File root = SPIFFS.open("/");
    File file = root.openNextFile();
    while (file) {
      String filename = String(file.name());
      if (!filename.endsWith(FFT_POSTFIX) && !filename.startsWith("model") && !filename.endsWith(LABEL_POSTFIX) && !filename.endsWith("identified")) {
        sampleFiles.push_back(filename);
      }
      file = root.openNextFile();
    }


    // === SORTOWANIE PO DARCIE W NAZWIE (DESCENDING) ===
    std::sort(sampleFiles.begin(), sampleFiles.end(), [](const String& a, const String& b) {
      return a.toInt() > b.toInt();  // większy timestamp = nowszy
    });


    for (const String& filename : sampleFiles) {
      String cleanName = filename;
      String timeStr = epochToLocalTime(cleanName.toInt());

      html += "<tr>";
      html += "<td>" + timeStr + "</td>";
      html += "<td>" + cleanName + " (<a href=/file?name=" + cleanName + "_fft>FFT</a>)</td>";
      html += "<td><audio controls style='width:300px;' src='/file?name=" + cleanName + "'&type=wav></audio></td>";
      html += "<td><form action='/label' method='get'>";
      html += "<input type='hidden' name='name' value='" + cleanName + "'>";
      String strKlasyfikacja = GetLabelForSample(cleanName);
      html += "<input type='text' name='label' placeholder='IDENTIFY' value='" + strKlasyfikacja + "'>";
      html += "<input type='submit' value='Save'></form></td>";
      html += "<td><a href='/file?name=" + filename + "' download&type=wav>Download</a> ";
      html += "<a href='/delete?name=" + filename + "'>Delete</a></td>";
      html += "</tr>";
    }
    html += "</table>";
    html += "</main>";


    // === FOOTER ===
    html += "<footer>";
    html += "<div style='display: flex; align-items: center; gap: 20px; flex-wrap: wrap;'>";

    // Formularz treningu
    html += "<form action='/train' method='get' style='display: flex; align-items: center; gap: 10px;'>";

    // Dropdown do wyboru liczby epok
    html += "<label for='epochs' style='color:#ccc;'>Epochs count:</label>";
    html += "<select name='epochs' id='epochs'>";
    html += "<option value='10'>10</option>";
    html += "<option value='50'>50</option>";
    html += "<option value='100' selected>100</option>";
    html += "<option value='200'>200</option>";
    html += "<option value='500'>500</option>";
    html += "</select>";
    html += "<input type='text' value='0.01' name='lr'>";
    html += "<input type='submit' value='Train Model'>";
    html += "</form>";

    // === UPLOAD FORM ===
    html += "<form action='/upload' method='POST' enctype='multipart/form-data' style='display: flex; align-items: center; gap: 10px;'>";
    html += "<label for='uploadfile' style='color:#ccc;'>📤 Upload file:</label>";
    html += "<input type='file' name='uploadfile' id='uploadfile'>";
    html += "<input type='submit' value='Upload'>";
    html += "</form>";

    // Linki + statystyki
    html += "<div style='display: flex; align-items: center; gap: 10px; font-size: 14px; flex-wrap: wrap;'>";
    html += "</div>";
    html += "</div>";
    html += "</footer>";
    html += "</body></html>";
    sampleFiles.clear();
    http.send(200, "text/html", html);
  });


  // Endpoint do zapisania labelki
  http.on("/label", []() {
    if (!http.hasArg("name") || !http.hasArg("label")) {
      http.send(400, "text/plain", "[ERROR] Missing parameters");
      return;
    }
    String name = http.arg("name");
    String label = http.arg("label");

    File fLabel = SPIFFS.open("/" + name + "_label", "w");
    fLabel.print(label);
    fLabel.close();

    http.sendHeader("Location", "/", true);
    http.send(302, "text/plain", "");
  });

  // Endpoint do treningu - na razie tylko symulacja treningu
  http.on("/train", []() {
    String strEpochs = http.arg("epochs");
    if (strEpochs == "")
      strEpochs = "100";


    String strLearnRate = http.arg("lr");
    if (strLearnRate == "")
      strLearnRate = "0.01";

    Training(strLearnRate.toDouble(), strEpochs.toInt());

    http.sendHeader("Location", "/", true);
    http.send(302, "text/plain", "");
  });

  // Obsługa pobierania pliku
  http.on("/file", []() {
    String strType = "text/plain";

    if (!http.hasArg("name")) {
      http.send(400, "text/plain", "Missing file name");
      return;
    }

    //
    String strRequestedType = http.arg("type");
    if (strRequestedType == "wav") strType = "audio/wav";

    String filename = "/" + http.arg("name");
    File file = SPIFFS.open(filename, "r");
    if (!file) {
      http.send(404, "text/plain", "File not found");
      return;
    }
    http.streamFile(file, strType);
    file.close();
  });

  // Endpoint do usuwania pliku
  http.on("/delete", []() {
    if (!http.hasArg("name")) {
      http.send(400, "text/plain", "Missing file name");
      return;
    }
    String filename = "/" + http.arg("name");
    if (SPIFFS.exists(filename)) {
      SPIFFS.remove(filename);

      // Usuwamy odpowiadający plik FFT, który powinien mieć nazwę: <filename> + "_fft"
      String fftFilename = filename + "_fft";
      if (SPIFFS.exists(fftFilename)) {
        SPIFFS.remove(fftFilename);
      }
      http.sendHeader("Location", "/", true);
      http.send(302, "text/plain", "");
    } else {
      http.send(404, "text/plain", "File not found");
    }
  });


  // Endpoint do usuwania modelu
  http.on("/delmodel", []() {
    SPIFFS.remove(MODEL_FILE);
    http.sendHeader("Location", "/", true);
    http.send(302, "text/plain", "");
    ESP.restart();
  });


  // Endpoint do usuwania modelu
  http.on("/autotrain", []() {
    bAutoTrain = !bAutoTrain;
    http.sendHeader("Location", "/", true);
    http.send(302, "text/plain", "");
  });
  // Endpoint do usuwania pliku
  http.on("/deleteallsamples", []() {
    // Przeglądanie plików w SPIFFS i wylistowanie tylko plików .wav
    File root = SPIFFS.open("/");
    File file = root.openNextFile();

    while (file) {
      String filename = String(file.name());
      // Nie kasuj
      if (filename.startsWith("model") || filename.startsWith("identified")) {
      } else {
        SPIFFS.remove("/" + filename);
      }
      file = root.openNextFile();
    }
    http.sendHeader("Location", "/", true);
    http.send(302, "text/plain", "");
  });


  // Endpoint w ktorym zczytujemy sume modułów w pasmie na potrzeby wyswietlania na html
  http.on("/level", []() {
    http.send(200, "text/plain", String(iAudioLevel));
  });


  http.on(
    "/upload", HTTP_POST, [&]() {
      http.send(200);
    },
    [&]() {
      HTTPUpload& uploadfile = http.upload();
      if (uploadfile.status == UPLOAD_FILE_START) {

        fFileUpload = SPIFFS.open("/" + uploadfile.filename, "w");
      } else if (uploadfile.status == UPLOAD_FILE_WRITE) {
        fFileUpload.write(uploadfile.buf, uploadfile.currentSize);
      } else if (uploadfile.status == UPLOAD_FILE_END) {
        fFileUpload.close();  // Close the file again
        http.sendHeader("Location", "/", true);
        http.send(302, "text/plain", "");
      }
    });
  http.on("/boot", [&]() {
    ESP.restart();
  });

  http.begin();

  configTime(gmtOffset_sec, daylightOffset_sec, ntpServer);

  initModel();
  loadModel(fLastTrainingError);

  // Allow OTA for easier uploads
  ArduinoOTA
    .onStart([]() {
      String type;
      if (ArduinoOTA.getCommand() == U_FLASH)
        type = "sketch";
      else  // U_SPIFFS
        type = "filesystem";
    })
    .onEnd([]() {

    })
    .onProgress([](unsigned int progress, unsigned int total) {
      yield();
    })
    .onError([](ota_error_t error) {
      ESP.restart();
    });
  ArduinoOTA.setHostname("BIRDIE");
  ArduinoOTA.begin();


  // Ustawienie mDNS
  if (MDNS.begin("esp32.birdie")) {
    Serial.println("mDNS ON");
  }


  pixels.setPixelColor(0, 0, 0, 0);
  pixels.show();
}

void loop() {
  M5.update();
  totalBytes = SPIFFS.totalBytes();
  usedBytes = SPIFFS.usedBytes();
  freeBytes = totalBytes - usedBytes;


  // NTP
  tm local_tm;
  if (!getLocalTime(&local_tm)) {
    timeSinceEpoch = 0;
    Serial.println("[INFO] NTP Failed");
  } else {
    timeSinceEpoch = mktime(&local_tm);
  }

  // Czytamy z mikrofonu
  size_t bytesRead = 0;
  i2s_read(SPEAK_I2S_NUMBER, (char*)audioBuffer, DATA_SIZE, &bytesRead, (10 / portTICK_PERIOD_MS));

  if (bytesRead > 0) {
    int16_t* samples = (int16_t*)audioBuffer;
    int numSamples = bytesRead / 2;
    int peak = 0;
    for (int i = 0; i < numSamples; i++) {
      int value = abs(samples[i]);
      if (value > peak) {
        peak = value;
      }
    }
    iAudioLevel = peak;

    // Jeśli wykryto sygnał o wysokiej amplitudzie i nie jesteśmy już w trakcie nagrywania
    if (peak > THRESHOLD && !recording) {
      recording = true;

      // Nagraj próbkę
      recordSegment(recordBuffer, RECORD_BUFFER_SIZE);

      double fftMagnitudes[INPUT_SIZE / 2];

      // Procesujemy pelne INPUT_SIZE
      int peakMagnitude = processAudio(fftMagnitudes, INPUT_SIZE);


      // Przykład detekcji
      if (peakMagnitude >= THRESHOLD) {

        pixels.setPixelColor(0, 0, 255, 0);
        pixels.show();

        // Przekazujemy magnitudy do modelu perceptronu (w predictLabel):
        float fPewnosc;
        String prediction = predictLabel(fftMagnitudes, INPUT_SIZE / 2, 0.8, fPewnosc);

        // Nie zna klasy
        if (prediction == "") {
          Serial.println("[INFO] Can't identify, storing sample for further labeling and trainings");
        } else {
          Serial.println("[INFO] Identified as '" + prediction + "' with " + String(fPewnosc) + "% of success");
        }


        if (freeBytes > SPIFFS_MIN_SIZE) {

          // Zapisz probke (sample WAV)
          char filename[32];
          sprintf(filename, "/%d", timeSinceEpoch);
          saveWavFile(filename, recordBuffer, RECORD_BUFFER_SIZE);

          // Zapisujemy nowe dane FFT do pliku:
          char fftFilename[40];
          sprintf(fftFilename, "%s_fft", filename);
          saveFFTData(fftFilename, fftMagnitudes, INPUT_SIZE / 2);

          // Dodaj label jesli rozpoznany
          if (prediction != "") {
            File fLabel = SPIFFS.open(String(filename) + "_label", "w");
            fLabel.print(prediction);
            fLabel.close();
          }
        }

        pixels.setPixelColor(0, 0, 0, 0);
        pixels.show();
      }

      recording = false;
    }
  }

  if (WiFi.status() != WL_CONNECTED) {
    ESP.restart();
  }




  // Sprawdzamy, czy jest 3:10 i czy zadanie nie zostało jeszcze wykonane
  if (local_tm.tm_hour == 3 && local_tm.tm_min == 10) {

    // Tutaj umieść kod, który ma się wykonać
    Serial.println("[INFO] Auto training...");

    Training(0.01, 500);
    delay(60000);  // Aby nie wpasc w ta sama minute
  }


  http.handleClient();
  ArduinoOTA.handle();
}
