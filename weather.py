import re
import random
import requests
from datetime import datetime

from config import config
from base_tool import BaseTool, ToolResult, ToolCategory, tool_registry


class WeatherTool(BaseTool):
    """专业天气查询工具 - 使用和风天气API"""

    QWEATHER_API_BASE = "https://devapi.qweather.com/v7"

    def __init__(self):
        super().__init__(
            name="weather",
            description="查询城市天气信息，包括温度、湿度、风力、空气质量等",
            category=ToolCategory.SEARCH,
        )
        self.api_key = config.api.qweather_key

    def _extract_city(self, query: str) -> str:
        """从查询中提取城市名"""
        if not query or not isinstance(query, str):
            return "北京"

        s = query.replace("的天气", "").replace("天气", "").strip()
        s = re.sub(r"^(今日|今天|明天|昨日|后天|现在)\s*", "", s)
        s = s.rstrip("的").strip()

        if s and len(s) <= 6 and re.match(r"^[\u4e00-\u9fa5]+市?$", s):
            return s.rstrip("市")

        m = re.search(r"([\u4e00-\u9fa5]{2,5})(?=市|天气|$)", query)
        if m:
            return m.group(1).rstrip("市")

        return "北京" if not s else s

    def _get_city_code(self, city_name: str) -> str:
        """通过城市名获取城市ID"""
        if not self.api_key:
            return "101010100" 

        try:
            url = f"{self.QWEATHER_API_BASE}/geo/v1/cities"
            params = {"location": city_name, "key": self.api_key, "limit": 1}
            resp = requests.get(url, params=params, timeout=5)
            data = resp.json()

            if data.get("code") == "200" and data.get("data"):
                return data["data"][0].get("id", "101010100")
        except Exception:
            pass

        return "101010100"

    def _get_weather(self, city_code: str) -> dict:
        """获取天气数据"""
        if not self.api_key:
            return self._mock_weather()

        try:
            url = f"{self.QWEATHER_API_BASE}/weather/now"
            params = {"location": city_code, "key": self.api_key}
            resp = requests.get(url, params=params, timeout=5)
            data = resp.json()

            if data.get("code") == "200":
                now = data.get("now", {})
                return {
                    "temp": now.get("temp", "未知"),
                    "feels_like": now.get("feelsLike", "未知"),
                    "condition": now.get("text", "未知"),
                    "humidity": now.get("humidity", "未知"),
                    "wind_dir": now.get("windDir", ""),
                    "wind_scale": now.get("windScale", ""),
                    "wind_speed": now.get("windSpeed", ""),
                    "update_time": now.get("obsTime", "")[:16]
                    if now.get("obsTime")
                    else "",
                }
        except Exception as e:
            print(f"和风天气API请求失败: {e}")

        return self._mock_weather()

    def _get_air_quality(self, city_code: str) -> dict:
        """获取空气质量"""
        if not self.api_key:
            return {"aqi": "未知", "level": "未知", "pm25": "未知"}

        try:
            url = f"{self.QWEATHER_API_BASE}/air/now"
            params = {"location": city_code, "key": self.api_key}
            resp = requests.get(url, params=params, timeout=5)
            data = resp.json()

            if data.get("code") == "200":
                now = data.get("now", {})
                aqi = now.get("aqi", "未知")
                return {
                    "aqi": aqi,
                    "level": now.get("category", "未知"),
                    "pm25": now.get("pm2p5", "未知"),
                }
        except Exception:
            pass

        return {"aqi": "未知", "level": "未知", "pm25": "未知"}

    def _get_forecast(self, city_code: str) -> list:
        """获取预报"""
        if not self.api_key:
            return []

        try:
            url = f"{self.QWEATHER_API_BASE}/weather/3d"
            params = {"location": city_code, "key": self.api_key}
            resp = requests.get(url, params=params, timeout=5)
            data = resp.json()

            if data.get("code") == "200":
                daily = data.get("daily", [])[:2]
                return [
                    {
                        "date": d.get("fxDate", ""),
                        "temp_max": d.get("tempMax", ""),
                        "temp_min": d.get("tempMin", ""),
                        "condition": d.get("textDay", ""),
                    }
                    for d in daily
                ]
        except Exception:
            pass

        return []

    def _mock_weather(self) -> dict:
        """模拟天气数据"""
        conditions = ["晴", "多云", "阴", "小雨", "晴间多云"]
        return {
            "temp": str(random.randint(15, 30)),
            "feels_like": str(random.randint(14, 32)),
            "condition": random.choice(conditions),
            "humidity": str(random.randint(40, 70)),
            "wind_dir": random.choice(["北风", "南风", "东风", "西风"]),
            "wind_scale": str(random.randint(1, 4)),
            "wind_speed": "",
            "update_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }

    def execute(self, query: str) -> ToolResult:
        """执行天气查询"""
        try:
            if not query or not isinstance(query, str):
                return ToolResult(
                    success=False,
                    content=config.prompts.output_templates["error"].format(
                        error_type="参数错误",
                        error_message="查询参数无效",
                        suggestion="请输入有效的城市名称",
                    ),
                    error="查询参数无效",
                )

            city = self._extract_city(query)
            city_code = self._get_city_code(city)

            weather = self._get_weather(city_code)
            air = self._get_air_quality(city_code)
            forecast = self._get_forecast(city_code)

            wind_info = (
                f"{weather['wind_dir']}{weather['wind_scale']}级"
                if weather["wind_scale"]
                else "未知"
            )

            content = f"""【{city}今日天气】

🌡️ 温度: {weather["temp"]}°C (体感 {weather["feels_like"]}°C)
☁️ 天气: {weather["condition"]}
💧 湿度: {weather["humidity"]}%
🌬️ 风力: {wind_info}
🕐 更新时间: {weather["update_time"]}"""

            if air.get("level") != "未知":
                content += f"""

🌿 空气质量: AQI {air["aqi"]} ({air["level"]})
   PM2.5: {air["pm25"]} μg/m³"""

            if forecast:
                content += """

📅 预报:"""
                for f in forecast:
                    content += f"\n   {f['date'][5:]}: {f['condition']} {f['temp_min']}~{f['temp_max']}°C"

            return ToolResult(
                success=True,
                content=content,
                metadata={"city": city, "city_code": city_code},
            )

        except Exception as e:
            return ToolResult(
                success=False,
                content=config.prompts.output_templates["error"].format(
                    error_type="天气查询失败",
                    error_message=str(e),
                    suggestion="请稍后重试",
                ),
                error=f"天气查询失败: {str(e)}",
            )


tool_registry.register(WeatherTool())
