#pragma once
#include <iostream>
#include <string>
#include <memory>
#include <time.h>
#include <chrono>
#include "spdlog/spdlog.h"
#include "spdlog/async.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/rotating_file_sink.h"

#include <xrtailor/core/Global.hpp>

namespace XRTailor {

#ifdef _WIN32
#define LOCALTIME(time_ptr, result_ptr) localtime_s((result_ptr), (time_ptr))
#else
#define LOCALTIME(time_ptr, result_ptr) localtime_r((time_ptr), (result_ptr))
#endif

static inline int ToDateInt(const tm& p) {
  return (1900 + p.tm_year) * 10000 + (p.tm_mon + 1) * 100 + p.tm_mday;
}

static inline int ToTimeInt(const tm& p) {
  return p.tm_hour * 10000 + p.tm_min * 100 + p.tm_sec;
}

static inline void NowDateTimeToInt(int& out_date, int& out_time) {
  time_t now;
  time(&now);
  tm p;
  LOCALTIME(&now, &p);

  out_date = ToDateInt(p);
  out_time = ToTimeInt(p);
}

class Logger {
 public:
  static Logger* GetInstance() {
    static Logger xlogger;
    return &xlogger;
  }

  std::shared_ptr<spdlog::logger> GetLogger() { return logger_; }

  void SwitchLogLevel(int level) {
    auto log_level = static_cast<spdlog::level::level_enum>(level);
    logger_->set_level(log_level);
    logger_->flush_on(log_level);
  }

 private:
  // make constructor private to avoid outside instance
  Logger() {
    // hardcode log path
    const std::string log_dir =
        Global::engine_config.log_path;  // should create the folder if not exist
    const std::string logger_name_prefix = "XRTailor_";

    // decide print to console or log file
    bool print_console = true;
    bool print_file = true;
    // decide the log level
    int level = Global::engine_config.log_level;

    try {
      // logger name with timestamp
      int date, time;
      NowDateTimeToInt(date, time);
      const std::string logger_name =
          logger_name_prefix + std::to_string(date) + "_" + std::to_string(time);
      // see https://github.com/gabime/spdlog/wiki/2.-Creating-loggers#creating-loggers-with-multiple-sinks for more details
      std::vector<spdlog::sink_ptr> sinks;
      if (print_console) {
        sinks.push_back(std::make_shared<spdlog::sinks::stdout_color_sink_st>());
      }
      if (print_console) {
        sinks.push_back(std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
            log_dir + "/" + logger_name + ".log", 512 * 1024 * 1024,
            1000));  // multi part log files, with every part 512MB, max 1000 files
      }
      logger_ = std::make_shared<spdlog::logger>(logger_name, begin(sinks), end(sinks));

      // custom format
      // see https://github.com/gabime/spdlog/wiki/3.-Custom-formatting#pattern-flags for more details
      logger_->set_pattern(
          "%Y-%m-%d %H:%M:%S [%^%l%$] [%s::%#] %v");  // with timestamp, filename and line number

      SwitchLogLevel(level);
    } catch (const spdlog::spdlog_ex& ex) {
      std::cout << "Log initialization failed: " << ex.what() << std::endl;
    }
  }

  ~Logger() {
    spdlog::drop_all();  // must do this
  }

  Logger(const Logger&) = delete;
  Logger& operator=(const Logger&) = delete;

 private:
  std::shared_ptr<spdlog::logger> logger_;
};

// use embedded macro to support file and line number
#define LOG_TRACE(...) \
  SPDLOG_LOGGER_CALL(Logger::GetInstance()->GetLogger().get(), spdlog::level::trace, __VA_ARGS__)
#define LOG_DEBUG(...) \
  SPDLOG_LOGGER_CALL(Logger::GetInstance()->GetLogger().get(), spdlog::level::debug, __VA_ARGS__)
#define LOG_INFO(...) \
  SPDLOG_LOGGER_CALL(Logger::GetInstance()->GetLogger().get(), spdlog::level::info, __VA_ARGS__)
#define LOG_WARN(...) \
  SPDLOG_LOGGER_CALL(Logger::GetInstance()->GetLogger().get(), spdlog::level::warn, __VA_ARGS__)
#define LOG_ERROR(...) \
  SPDLOG_LOGGER_CALL(Logger::GetInstance()->GetLogger().get(), spdlog::level::err, __VA_ARGS__)
}  // namespace XRTailor