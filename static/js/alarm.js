// Fire Detection System - Alarm Sound
// Plays a beep-beep-beep alarm using the Web Audio API (no audio file needed)

function playAlarm() {
    function beep(freq, duration, delay) {
        setTimeout(() => {
            const ctx = new (window.AudioContext || window.webkitAudioContext)();
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.connect(gain);
            gain.connect(ctx.destination);
            osc.type = "square";
            osc.frequency.value = freq;
            gain.gain.value = 0.3;
            osc.start();
            setTimeout(() => { osc.stop(); ctx.close(); }, duration);
        }, delay);
    }
    beep(1000, 250, 0);
    beep(1000, 250, 400);
    beep(1000, 250, 800);
}
