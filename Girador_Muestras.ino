// ================================================================
// GIRADOR DE MUESTRAS
// Versión  2.6 del día 6 de marzo de 2026
// ================================================================

// Librerías
#include <LiquidCrystal_I2C.h>
#include <LiquidMenu.h>
#include <AccelStepper.h>

// ================================================================
// CONFIGURACIÓN DE HARDWARE
// ================================================================

#define STEP_PIN          12
#define DIR_PIN           13
#define ENCODER_PIN_CLK   2
#define ENCODER_PIN_DT    3
#define ENCODER_BUTTON    7
#define BOTON_HOME        8
#define BOTON_GIRO        9
#define BOTON_SAVE        10

const int STEPS_PER_REV  = 200;
const int MICRO_STEPS    = 16;
const int MAX_SPEED      = 1000;
const int ACCELERATION   = 5000;

// ================================================================
// OBJETOS PRINCIPALES
// ================================================================

LiquidCrystal_I2C lcd(0x27, 20, 4);
AccelStepper stepper(AccelStepper::DRIVER, STEP_PIN, DIR_PIN);

// ================================================================
// CLASE BUTTON — DEBOUNCE NO BLOQUEANTE
// ================================================================

class Button {
  private:
    int pin;
    unsigned long lastDebounceTime;
    bool lastButtonState;
    bool buttonState;
    const unsigned long debounceDelay = 50;

  public:
    Button(int p) : pin(p), lastDebounceTime(0), lastButtonState(HIGH), buttonState(HIGH) {
      pinMode(pin, INPUT_PULLUP);
    }

    bool isPressed() {
      bool reading = digitalRead(pin);

      // Si cambió el estado, reiniciar el temporizador
      if (reading != lastButtonState) {
        lastDebounceTime = millis();
      }

      if ((millis() - lastDebounceTime) > debounceDelay) {
        if (reading != buttonState) {
          buttonState = reading;
          if (buttonState == LOW) {
            return true;
          }
        }
      }

      lastButtonState = reading;
      return false;
    }
};

Button encoderBtn(ENCODER_BUTTON);
Button botonSave(BOTON_SAVE);
Button botonHome(BOTON_HOME);
Button botonGiro(BOTON_GIRO);

// ================================================================
// VARIABLES GLOBALES
// ================================================================

volatile long encoderPos   = 0;
volatile int lastStateCLK  = 0;

long currentAngle    = 0;
long homeAngle       = 0;
long anguloAcumulado = 0;

bool inGiroLibre     = false;
bool inAnguloPreest  = false;
bool homingCompleted = false;

// ================================================================
// MENÚ
// ================================================================

LiquidLine headerPantalla1(0, 0, "---Menu Principal---");
LiquidLine linea1(1, 1, "Giro Libre");
LiquidLine linea2(1, 2, "Giro Fijo");
LiquidScreen pantalla1(headerPantalla1, linea1, linea2);

LiquidLine headerPantalla2(0, 0, "-----Giro Libre-----");
LiquidLine lineaAngulo(0, 1, "Posicion Rel.: ", currentAngle);
LiquidScreen pantalla2(headerPantalla2, lineaAngulo);

LiquidLine headerPantalla3(0, 0, "----Giro Manual----");
LiquidLine lineaAnguloPreest(0, 1, "Posicion Rel.: ", currentAngle);
LiquidLine lineaAngAcumulado(0, 2, "Posicion Abs.: ", anguloAcumulado);
LiquidScreen pantalla3(headerPantalla3, lineaAnguloPreest, lineaAngAcumulado);

LiquidMenu menu(lcd, pantalla1, pantalla2, pantalla3);

// ================================================================
// DECLARACIÓN DE FUNCIONES
// ================================================================

void handleMenuNavigation();
void handleGiroLibre();
void handleAnguloPreest();
void saveHomeAngle();
void returnToHome();
void resetEncoder();
void returnToMainMenu();
void updateEncoderPosition();
void encoderISR_A();
void fn_giro_libre();
void fn_angulo_preest();

// ================================================================
// SETUP
// ================================================================

void setup() {
  Serial.begin(9600);

  lcd.init();
  lcd.backlight();

  stepper.setMaxSpeed(MAX_SPEED);
  stepper.setAcceleration(ACCELERATION);

  attachInterrupt(digitalPinToInterrupt(ENCODER_PIN_CLK), encoderISR_A, CHANGE);

  linea1.set_focusPosition(Position::LEFT);
  linea2.set_focusPosition(Position::LEFT);
  linea1.attach_function(1, fn_giro_libre);
  linea2.attach_function(1, fn_angulo_preest);
  menu.add_screen(pantalla1);
  menu.add_screen(pantalla2);
  menu.add_screen(pantalla3);
  pantalla1.set_displayLineCount(3);
  menu.set_focusedLine(1);
  menu.update();

  Serial.println("Sistema iniciado correctamente");
}

// ================================================================
// LOOP
// ================================================================

void loop() {
  updateEncoderPosition();

  if (inGiroLibre) {
    handleGiroLibre();
  } 
  else if (inAnguloPreest) {
    handleAnguloPreest();
  } 
  else {
    handleMenuNavigation();
  }
}

// ================================================================
// INTERRUPCIONES ISR ENCODER
// ================================================================

void encoderISR_A() {
  int currentStateCLK = digitalRead(ENCODER_PIN_CLK);
  int currentStateDT  = digitalRead(ENCODER_PIN_DT);

  if (currentStateCLK != lastStateCLK) {
    //Giro horario
    if (currentStateDT != currentStateCLK) {
      encoderPos++;
    } 
    //Giro antihorario
    else {
      encoderPos--;
    }
  }

  lastStateCLK = currentStateCLK;
}

// ================================================================
// FUNCIONES UTILES, CREADAS PARA ENCAPSULAR CODIGO
// ================================================================

void updateEncoderPosition() {
  noInterrupts();
  long tempPos = encoderPos;
  interrupts();
  currentAngle = tempPos;
}

void resetEncoder() {
  noInterrupts();
  encoderPos = 0;
  interrupts();
  currentAngle = 0;
}

void returnToMainMenu() {
  menu.change_screen(1);
  menu.set_focusedLine(0);
  menu.update();
}

// ================================================================
// NAVEGACIÓN DEL MENÚ
// ================================================================

void handleMenuNavigation() {
  static long lastMenuPos = 0;
  static unsigned long lastMenuUpdate = 0;
  const unsigned long menuUpdateDelay = 200;

  noInterrupts();
  long tempPos = encoderPos;
  interrupts();

  if (millis() - lastMenuUpdate > menuUpdateDelay) {
    if (tempPos > lastMenuPos) {
      //Mueve hacia abajo
      menu.switch_focus(true);
      menu.update();
      lastMenuPos = tempPos;
      lastMenuUpdate = millis();
    } else if (tempPos < lastMenuPos) {
      //Mueve hacia arriba
      menu.switch_focus(false);
      menu.update();
      lastMenuPos = tempPos;
      lastMenuUpdate = millis();
    }
  }

  if (encoderBtn.isPressed()) {
    menu.call_function(1);
  }
}

// ================================================================
// MODO GIRO LIBRE
// ================================================================

void handleGiroLibre() {
  stepper.run();

  long targetPosition = (currentAngle * STEPS_PER_REV * MICRO_STEPS) / 360;
  stepper.moveTo(targetPosition);
  while (stepper.distanceToGo() != 0) {
    stepper.run();
  }

  lcd.setCursor(0, 0);
  lcd.print("-----Giro Libre-----");
  lcd.setCursor(0, 1);
  lcd.print("Posicion Rel.: ");
  lcd.print(currentAngle);
  lcd.print("   ");

  if (encoderBtn.isPressed()) {
    stepper.stop();
    saveHomeAngle();
    returnToHome();
    inGiroLibre = false;
    lcd.clear();
    returnToMainMenu();
  }

  if (botonSave.isPressed()) {
    saveHomeAngle();
  }

  if (botonHome.isPressed()) {
    returnToHome();
  }
}

// ================================================================
// MODO GIRO MANUAL
// ================================================================

void handleAnguloPreest() {
  stepper.run();

  lcd.setCursor(0, 0);
  lcd.print("----Giro  Manual----");
  lcd.setCursor(0, 1);
  lcd.print("Posicion Rel.: ");
  lcd.print(currentAngle);
  lcd.print("   ");
  lcd.setCursor(0, 2);
  lcd.print("Posicion Abs.: ");
  lcd.print(anguloAcumulado);
  lcd.print("   ");

  if (encoderBtn.isPressed()) {
    stepper.stop();
    saveHomeAngle();
    returnToHome();
    inAnguloPreest = false;
    anguloAcumulado = 0;
    lcd.clear();
    returnToMainMenu();
  }

  if (botonSave.isPressed()) {
    saveHomeAngle();
  }

  if (botonHome.isPressed()) {
    returnToHome();
  }

  if (botonGiro.isPressed()) {
    long targetPosition = (currentAngle * STEPS_PER_REV * MICRO_STEPS) / 360;
    stepper.move(targetPosition);

    //Movemos hasta que llegue a la posicón objetivo
    while (stepper.distanceToGo() != 0) {
      stepper.run();
    }
    anguloAcumulado += currentAngle;
    lcd.setCursor(0, 2);
    lcd.print("Posicion Abs.: ");
    lcd.print(anguloAcumulado);
    lcd.print("   ");
  }
}

// ================================================================
// FUNCIONES DE HOME
// ================================================================

void saveHomeAngle() {
  stepper.stop();
  long homePosition = stepper.currentPosition();
  homeAngle = (homePosition * 360) / (STEPS_PER_REV * MICRO_STEPS);

  Serial.print("Home guardado: ");
  Serial.println(homeAngle);

  lcd.setCursor(0, 3);
  lcd.print("Home guardado       ");
  delay(500);
  lcd.clear();
}

void returnToHome() {
  long homePosition = (homeAngle * STEPS_PER_REV * MICRO_STEPS) / 360;

  Serial.print("Moviendo a home: ");
  Serial.println(homePosition);

  lcd.setCursor(0, 3);
  lcd.print("Moviendo a home...  ");

  stepper.moveTo(homePosition);
  while (stepper.distanceToGo() != 0) {
    stepper.run();
  }

  stepper.stop();
  stepper.setCurrentPosition(0);
  resetEncoder();
  anguloAcumulado = 0;
  homeAngle       = 0;
  homingCompleted = true;

  lcd.setCursor(0, 3);
  lcd.print("Home alcanzado      ");
  delay(750);
  lcd.clear();
}

// ================================================================
// CALLBACKS DEL MENÚ
// ================================================================

void fn_giro_libre() {
  resetEncoder();
  inGiroLibre    = true;
  homingCompleted = false;
  menu.change_screen(2);
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("-----Giro Libre-----");
  lcd.setCursor(0, 1);
  lcd.print("Posicion Rel.: 0    ");
}

void fn_angulo_preest() {
  resetEncoder();
  anguloAcumulado = 0;
  inAnguloPreest  = true;
  menu.change_screen(3);
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("----Giro  Manual----");
  lcd.setCursor(0, 1);
  lcd.print("Posicion Rel.: 0    ");
  lcd.setCursor(0, 2);
  lcd.print("Posicion Abs.: 0    ");
}
