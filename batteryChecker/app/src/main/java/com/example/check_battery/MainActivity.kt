package com.example.check_battery

import android.content.*
import android.os.*
import android.provider.Settings
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.example.check_battery.ui.theme.Check_BatteryTheme
import kotlinx.coroutines.*
import java.io.File
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.Executors
import kotlin.math.pow

private const val TAG = "PowerLogger"
private const val SAMPLE_MS = 2_000L          // Период выборки, 1 секунда
private const val CSV_NAME = "power_log.csv"  // Имя файла в Documents
private val TRACKED_CORES = mapOf(0 to 0, 1 to 7) // cluster → core
private const val BENCHMARK_DURATION_MIN = 60  // Длительность бенчмарка, минуты
private val BRIGHTNESS_LEVELS = listOf(1, 25, 50, 75, 100) // Уровни яркости, %
private val CPU_LOAD_LEVELS = listOf(0, 25, 50, 75, 100)   // Уровни нагрузки CPU, %
private var MAX_BRIGHTNESS = 2047                     // Максимальная яркость (можно изменить)

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Определяем максимальную яркость устройства
        try {
            val maxBrightness = File("/sys/class/backlight").listFiles()?.firstOrNull()
                ?.let { File(it, "max_brightness").readText().trim().toInt() } ?: 2047
            MAX_BRIGHTNESS = maxBrightness
        } catch (e: Exception) {
            Log.e(TAG, "Failed to read max brightness: ${e.message}")
        }
        setContent { Check_BatteryTheme { CpuDisplayBatteryLogger() } }
    }
}

@Composable
fun CpuDisplayBatteryLogger() {
    val ctx = LocalContext.current
    if (!Settings.System.canWrite(ctx)) {
        ctx.startActivity(Intent(Settings.ACTION_MANAGE_WRITE_SETTINGS))
    }
    val pm = remember { ctx.getSystemService(PowerManager::class.java) }
    val bm = remember { ctx.getSystemService(BatteryManager::class.java) }
    val scope = rememberCoroutineScope()

    /* prev-снимки */
    var prevTis by remember { mutableStateOf<Map<Int, Map<Int, Long>>?>(null) } // cluster → freq→ms
    var prevCid by remember { mutableStateOf<List<Map<String, Long>>?>(null) }  // core → state→ms

    /* CSV-файл */
    val csvFile = remember {
        val docsDir = ctx.getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS)!!
        docsDir.mkdirs()
        File(docsDir, CSV_NAME)
    }
    var headerWritten by remember { mutableStateOf(false) }
    var headerCols by remember { mutableStateOf(listOf<String>()) }

    /* UI-лог */
    val uiLog = remember { mutableStateListOf<String>() }

    /* Состояние бенчмарка */
    var benchmarkRunning by remember { mutableStateOf(false) }
    var currentBrightnessPct by remember { mutableStateOf(0) }
    var currentCpuLoadPct by remember { mutableStateOf(0) }
    var currentStage by remember { mutableStateOf("Idle, Idle") }

    /* Функция логирования */
    suspend fun logData(cores: Int, nowTis: Map<Int, Map<Int, Long>>, nowCid: List<Map<String, Long>>) {
        val curUa = bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW)        // µA
        val qUah = bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER)      // µAh
        val soc = bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY)            // %
        val voltMv = ctx.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
            ?.getIntExtra(BatteryManager.EXTRA_VOLTAGE, -1) ?: -1
        val battTemp = getBatteryTemperature(ctx)
        val cpuTemp = getCpuTemperature()
        val dispOn = pm.isInteractive
        val brightLin = getLinearBrightness(ctx)

        /* ---------- warm-up? ---------- */
        if (prevTis == null || prevCid == null) {
            prevTis = nowTis
            prevCid = nowCid
            if (!headerWritten) {
                headerCols = buildHeader(nowTis)
                csvFile.delete() // Удаляем старый CSV
                csvFile.appendText(headerCols.joinToString(",") + "\n")
                headerWritten = true
            }
            return
        }

        /* ---------- Проверка заголовков CSV ---------- */
        if (csvFile.exists()) {
            val currentHeader = csvFile.readLines().firstOrNull()?.split(",") ?: emptyList()
            if (currentHeader != headerCols) {
                csvFile.delete()
                csvFile.appendText(headerCols.joinToString(",") + "\n")
                headerWritten = true
            }
        }

        /* ---------- расчёт Δ ---------- */
        val timeStamp = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault()).format(Date())
        val csvRow = mutableListOf<String>()
        csvRow += timeStamp
        csvRow += SAMPLE_MS.toString()
        csvRow += "%.3f".format(curUa / 1000.0)
        csvRow += "%.3f".format(voltMv / 1000.0)
        csvRow += qUah.toString()
        csvRow += soc.toString()
        csvRow += "%.1f".format(battTemp)
        csvRow += "%.1f".format(cpuTemp)
        csvRow += if (dispOn) "1" else "0"
        csvRow += "%.3f".format(brightLin)
        csvRow += currentBrightnessPct.toString()
        csvRow += currentCpuLoadPct.toString()

        val sbUi = StringBuilder("\n[${timeStamp.substring(11)}]\n")
        sbUi.append("BAT ⎓${curUa/1000} mA  V=${voltMv/1000.0} V  Q=${qUah} µAh  SoC=$soc%  Temp=${"%.1f".format(battTemp)}°C\n")
        sbUi.append("CPU Temp=${"%.1f".format(cpuTemp)}°C\n")
        sbUi.append("DISP ").append(if (dispOn) "ON " else "OFF").append("  Bright=${"%.2f".format(brightLin * 100)}%\n")
        sbUi.append("Stage: Brightness_${currentBrightnessPct}%, CPULoad_${currentCpuLoadPct}%\n")
        sbUi.append("CPU per-core (Δ${SAMPLE_MS/1000}s):\n")

        for (core in 0 until cores) {
            val dIdle = deltaMs(nowCid[core], prevCid!![core]).coerceAtMost(SAMPLE_MS)
            val dAct = (SAMPLE_MS - dIdle).coerceAtLeast(0)
            val load = if (dAct + dIdle > 0) dAct * 100 / (dAct + dIdle) else 0

            csvRow += dIdle.toString()

            val cluster = when (core) {
                in 0..5 -> 0
                else -> 1
            }
            val dFreq = if (core in TRACKED_CORES.values) {
                deltaFreq(nowTis[cluster] ?: emptyMap(), prevTis!![cluster] ?: emptyMap())
            } else emptyMap()
            val topFreq = dFreq.maxByOrNull { it.value }?.key ?: -1
            sbUi.append("  cpu$core: act=${dAct} ms  idle=${dIdle} ms  load=$load%")
            if (core in TRACKED_CORES.values) {
                sbUi.append("  topFreq=").append(if (topFreq > 0) "${topFreq/1000} MHz" else "?")
            }
            sbUi.append('\n')
        }

        TRACKED_CORES.keys.forEach { cluster ->
            headerCols.filter { it.startsWith("cluster${cluster}_") }.forEach { col ->
                val freq = col.substringAfter('_').substringBefore('_').toInt()
                csvRow += nowTis[cluster]?.let { deltaFreq(it, prevTis!![cluster] ?: emptyMap())[freq] }?.toString() ?: "0"
            }
        }

        csvFile.appendText(csvRow.joinToString(",") + "\n")
        Log.d(TAG, sbUi.toString())
        uiLog += sbUi.toString()

        prevTis = nowTis
        prevCid = nowCid
    }

    /* Запуск бенчмарка */
    fun startBenchmark() {
        scope.launch {
            benchmarkRunning = true
            val cores = File("/sys/devices/system/cpu")
                .listFiles { f -> Regex("cpu[0-9]+").matches(f.name) }?.size ?: 0
            runBenchmark(ctx, cores) { brightnessPct, cpuLoadPct ->
                currentBrightnessPct = brightnessPct
                currentCpuLoadPct = cpuLoadPct
                currentStage = "Brightness_${brightnessPct}%, CPULoad_${cpuLoadPct}%"
                uiLog += currentStage
            }
            benchmarkRunning = false
            currentBrightnessPct = 0
            currentCpuLoadPct = 0
            currentStage = "Idle, Idle"
            uiLog += currentStage
        }
    }

    LaunchedEffect(Unit) {
        val cores = File("/sys/devices/system/cpu")
            .listFiles { f -> Regex("cpu[0-9]+").matches(f.name) }?.size ?: 0
        while (true) {
            val nowTis = TRACKED_CORES.mapValues { readTimeInState(it.value) }
            val nowCid = List(cores) { readCpuidle(it) }
            logData(cores, nowTis, nowCid)
            delay(SAMPLE_MS)
        }
    }

    Surface(Modifier.fillMaxSize().padding(12.dp)) {
        Column(Modifier.verticalScroll(rememberScrollState())) {
            Text("⚡ Power Logger & Benchmark → $CSV_NAME", style = MaterialTheme.typography.headlineMedium)
            Spacer(Modifier.height(8.dp))
            Button(
                onClick = { if (!benchmarkRunning) startBenchmark() },
                enabled = !benchmarkRunning
            ) {
                Text("Start Benchmark ($BENCHMARK_DURATION_MIN min)")
            }
            Spacer(Modifier.height(8.dp))
            Text("Current Stage: $currentStage")
            Spacer(Modifier.height(8.dp))
            Text("Max Brightness: $MAX_BRIGHTNESS")
            Spacer(Modifier.height(8.dp))
        }
    }
}

/* ---------------- helper-функции ---------------- */

private fun buildHeader(tisSnap: Map<Int, Map<Int, Long>>): List<String> {
    val cols = mutableListOf(
        "timestamp",
        "sample_ms",
        "batt_current_mA",
        "batt_voltage_V",
        "batt_charge_uAh",
        "batt_soc_pct",
        "batt_temp_C",
        "cpu_temp_C",
        "display_on",
        "display_brightness_lin",
        "brightness_pct",
        "cpu_load_pct"
    )
    val cores = File("/sys/devices/system/cpu")
        .listFiles { f -> Regex("cpu[0-9]+").matches(f.name) }?.size ?: 0
    for (core in 0 until cores) {
        cols += "core${core}_idle_ms"
    }
    TRACKED_CORES.keys.sorted().forEach { cluster ->
        tisSnap[cluster]?.keys?.sorted()?.forEach { freq ->
            cols += "cluster${cluster}_${freq}_ms"
        }
    }
    return cols
}

private fun getLinearBrightness(ctx: Context): Float {
    val raw = Settings.System.getInt(ctx.contentResolver, Settings.System.SCREEN_BRIGHTNESS, 0)
    val norm = raw / MAX_BRIGHTNESS.toFloat().coerceAtLeast(1f)
    return norm.pow(2.2f)
}

private fun setBrightness(ctx: Context, level: Int) {
    try {
        Settings.System.putInt(ctx.contentResolver, Settings.System.SCREEN_BRIGHTNESS, level.coerceIn(0, MAX_BRIGHTNESS))
    } catch (e: Exception) {
        Log.e(TAG, "Failed to set brightness: ${e.message}")
    }
}

private fun readTimeInState(core: Int): Map<Int, Long> = try {
    File("/sys/devices/system/cpu/cpu$core/cpufreq/stats/time_in_state")
        .readLines()
        .mapNotNull { line ->
            val p = line.trim().split(Regex("\\s+"))
            val freq = p.getOrNull(0)?.toIntOrNull() ?: return@mapNotNull null
            val ticks = p.getOrNull(1)?.toLongOrNull() ?: return@mapNotNull null
            freq to ticks * 10
        }.toMap()
} catch (_: Exception) { emptyMap() }

private fun readCpuidle(core: Int): Map<String, Long> {
    return try {
        val dir = File("/sys/devices/system/cpu/cpu$core/cpuidle")
            .listFiles { d -> d.isDirectory && d.name.startsWith("state") } ?: return emptyMap()
        val raw = dir.associate { st ->
            val name = File(st, "name").readText().trim()
            val time = File(st, "time").readText().trim().toLong()
            name to time
        }
        val scale = if (raw.values.sum() > 1_000_000_000_000L) 1_000_000L else 1_000L
        raw.mapValues { it.value / scale }
    } catch (_: Exception) { emptyMap() }
}

private fun deltaMs(now: Map<String, Long>, prev: Map<String, Long>) =
    (now.keys + prev.keys).sumOf { k -> (now[k] ?: 0) - (prev[k] ?: 0) }.coerceAtMost(SAMPLE_MS)

private fun deltaFreq(now: Map<Int, Long>, prev: Map<Int, Long>): Map<Int, Long> =
    (now.keys + prev.keys).associateWith { f -> (now[f] ?: 0) - (prev[f] ?: 0) }

private fun getBatteryTemperature(ctx: Context): Float {
    val intent = ctx.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
    val temp = intent?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, -1) ?: -1
    return if (temp != -1) temp / 10.0f else -1f
}

private fun getCpuTemperature(): Float {
    try {
        val thermalZones = File("/sys/class/thermal")
            .listFiles { f -> f.isDirectory && f.name.startsWith("thermal_zone") } ?: return -1f
        for (zone in thermalZones) {
            val type = File(zone, "type").readText().trim()
            if (type.contains("cpu", ignoreCase = true)) {
                val temp = File(zone, "temp").readText().trim().toFloatOrNull() ?: return -1f
                return temp / 1000f
            }
        }
    } catch (e: Exception) {
        Log.e(TAG, "Error reading CPU temp: ${e.message}")
    }
    return -1f
}

/* ---------------- Бенчмарк ---------------- */

private suspend fun runBenchmark(
    ctx: Context,
    cores: Int,
    onStageChange: (Int, Int) -> Unit
) {
    val executor = Executors.newFixedThreadPool(cores)
    val totalDurationMs = BENCHMARK_DURATION_MIN * 60 * 1000L
    val totalStages = BRIGHTNESS_LEVELS.size * CPU_LOAD_LEVELS.size
    val stageDurationMs = (totalDurationMs / totalStages).coerceAtLeast(SAMPLE_MS)
    val brightnessLevels = BRIGHTNESS_LEVELS.map { (it * MAX_BRIGHTNESS / 100).toInt() }
    val cpuLoadLevels = CPU_LOAD_LEVELS.map { it / 100f }

    // Отключаем автоматическую яркость
    Settings.System.putInt(ctx.contentResolver, Settings.System.SCREEN_BRIGHTNESS_MODE, Settings.System.SCREEN_BRIGHTNESS_MODE_MANUAL)

    try {
        BRIGHTNESS_LEVELS.forEachIndexed { i, brightnessPct ->
            CPU_LOAD_LEVELS.forEachIndexed { j, cpuLoadPct ->
                val brightness = brightnessLevels[i]
                val cpuLoad = cpuLoadLevels[j]
                onStageChange(brightnessPct, cpuLoadPct)
                Log.d(TAG, "Starting stage: Brightness_${brightnessPct}%_CPULoad_${cpuLoadPct}%")

                // Устанавливаем яркость
                setBrightness(ctx, brightness)

                // Создаём нагрузку на CPU
                val tasks = (0 until cores).map { core ->
                    Runnable {
                        val startTime = System.currentTimeMillis()
                        while (System.currentTimeMillis() - startTime < stageDurationMs) {
                            if (cpuLoad > 0f) {
                                // Выполняем вычисления для нагрузки
                                val workDuration = (SAMPLE_MS * cpuLoad).toLong()
                                val sleepDuration = (SAMPLE_MS * (1 - cpuLoad)).toLong()
                                val cycleStart = System.currentTimeMillis()
                                while (System.currentTimeMillis() - cycleStart < workDuration) {
                                    // Интенсивные вычисления (Фибоначчи)
                                    var a = 0L
                                    var b = 1L
                                    for (k in 0 until 100_000) {
                                        val c = a + b
                                        a = b
                                        b = c
                                    }
                                }
                                Thread.sleep(sleepDuration.coerceAtLeast(1L))
                            } else {
                                Thread.sleep(SAMPLE_MS)
                            }
                        }
                    }
                }

                // Запускаем задачи
                tasks.forEach { executor.submit(it) }

                // Ждём завершения этапа
                delay(stageDurationMs)
            }
        }
    } finally {
        executor.shutdown()
        // Восстанавливаем яркость (50%)
        setBrightness(ctx, (MAX_BRIGHTNESS * 0.5).toInt())
    }
}