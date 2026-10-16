import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sqlalchemy import text

from db import engine

# securities.sector for KRX tickers holds the KOSDAQ market-tier/division name (e.g. "우량기업부"),
# not a real industry classification -- KRX's official Open API has no industry-classification
# endpoint (checked), and no well-maintained free source covers our ~3,700-ticker pool. One-time,
# LLM-assigned GICS-style sector for just the CURRENT universe (checked by hand against each
# company's actual business), as a stand-in until/unless a real data source is wired in. Only
# touches the `sector` column -- nothing else about securities or any other table.
SECTOR_OVERRIDES = {
    # KOSDAQ -- almost entirely semiconductor equipment/materials/components
    "010170": "Information Technology",  # 대한광통신 (optical fiber/telecom equipment)
    "024840": "Materials",  # 케이비아이메탈 (copper/brass strip for batteries/electronics)
    "031980": "Information Technology",  # 피에스케이홀딩스 (semiconductor equipment holding)
    "032580": "Information Technology",  # 피델릭스 (fabless memory IC)
    "032820": "Industrials",  # 우리기술 (industrial/nuclear plant automation)
    "036540": "Information Technology",  # 에스에프에이반도체 (semiconductor packaging/test)
    "036930": "Information Technology",  # 주성엔지니어링 (semiconductor/display equipment)
    "039030": "Information Technology",  # 이오테크닉스 (laser equipment for semiconductor/display)
    "043260": "Information Technology",  # 성호전자 (electronic components)
    "046970": "Industrials",  # 우리로 (optical/fire sensors)
    "058610": "Industrials",  # 에스피지 (precision gears/motors)
    "064290": "Information Technology",  # 인텍플러스 (semiconductor/battery inspection equipment)
    "064760": "Information Technology",  # 티씨케이 (semiconductor SiC parts)
    "067310": "Information Technology",  # 하나마이크론 (semiconductor packaging)
    "069540": "Information Technology",  # 빛과전자 (electronic connectors)
    "079650": "Information Technology",  # 서산 (electronic connectors)
    "080220": "Information Technology",  # 제주반도체 (fabless memory IC)
    "080580": "Information Technology",  # 오킨스전자 (semiconductor test sockets)
    "084370": "Information Technology",  # 유진테크 (semiconductor CVD equipment)
    "085910": "Information Technology",  # 네오티스 (semiconductor/display materials)
    "086520": "Materials",  # 에코프로 (EV battery cathode materials)
    "089970": "Information Technology",  # 브이엠 (semiconductor etching equipment)
    "092870": "Information Technology",  # 엑시콘 (semiconductor test equipment)
    "095340": "Information Technology",  # 아이에스시 (semiconductor test sockets)
    "095610": "Information Technology",  # 테스 (semiconductor CVD equipment)
    "098120": "Information Technology",  # 마이크로컨텍솔루션 (semiconductor test probes)
    "108490": "Industrials",  # 로보티즈 (robotics components)
    "131290": "Information Technology",  # 티에스이 (semiconductor test sockets/probe cards)
    "131970": "Information Technology",  # 두산테스나 (semiconductor OSAT test)
    "159010": "Information Technology",  # 아스플로 (semiconductor gas delivery systems)
    "219130": "Materials",  # 타이거일렉 (EV battery components)
    "222800": "Information Technology",  # 심텍 (PCB/semiconductor substrates)
    "240810": "Information Technology",  # 원익아이피에스 (semiconductor/display equipment)
    "252990": "Information Technology",  # 샘씨엔에스 (ceramic semiconductor substrates)
    "319660": "Information Technology",  # 피에스케이 (semiconductor etching/ashing equipment)
    "327260": "Information Technology",  # 알에프머트리얼즈 (semiconductor RF materials)
    "403870": "Information Technology",  # 에이치피에스피 (semiconductor annealing equipment)
    "417840": "Information Technology",  # 저스템 (semiconductor wafer handling equipment)
    "420770": "Information Technology",  # 기가비스 (semiconductor substrate inspection)
    "440110": "Information Technology",  # 파두 (fabless SSD controller IC)
    # KOSPI
    "000150": "Industrials",  # 두산 (conglomerate holding)
    "000500": "Industrials",  # 가온전선 (electric cables)
    "000660": "Information Technology",  # 에스케이하이닉스 (memory semiconductor)
    "000720": "Industrials",  # 현대건설 (construction)
    "000990": "Information Technology",  # DB하이텍 (semiconductor foundry)
    "001820": "Information Technology",  # 삼화콘덴서공업 (capacitors)
    "002990": "Industrials",  # 금호건설 (construction)
    "003160": "Information Technology",  # 디아이 (semiconductor burn-in test equipment)
    "005380": "Consumer Discretionary",  # 현대자동차 (automobiles)
    "005930": "Information Technology",  # 삼성전자 (semiconductors/electronics)
    "005935": "Information Technology",  # 삼성전자1우선주
    "006110": "Materials",  # 삼아알미늄 (aluminum foil)
    "006340": "Industrials",  # 대원전선 (electric cables)
    "006360": "Industrials",  # GS건설 (construction)
    "006400": "Information Technology",  # 삼성SDI (batteries/electronic materials)
    "007660": "Information Technology",  # 이수페타시스 (PCB)
    "007810": "Information Technology",  # 코리아써키트 (PCB)
    "009150": "Information Technology",  # 삼성전기 (MLCC/electronic components)
    "009155": "Information Technology",  # 삼성전기1우선주
    "010060": "Materials",  # 오씨아이홀딩스 (chemicals, polysilicon)
    "010120": "Industrials",  # LS ELECTRIC (electrical equipment)
    "010950": "Energy",  # S-Oil (oil refining)
    "011070": "Information Technology",  # LG이노텍 (electronic components/camera modules)
    "017670": "Communication Services",  # SK텔레콤 (telecom carrier)
    "017900": "Information Technology",  # 광전자 (optoelectronic components)
    "028050": "Industrials",  # 삼성E&A (engineering/construction)
    "028260": "Industrials",  # 삼성물산 (construction/trading conglomerate)
    "032830": "Financials",  # 삼성생명보험 (insurance)
    "034020": "Industrials",  # 두산에너빌리티 (power plant equipment)
    "034730": "Industrials",  # SK (holding company)
    "042700": "Information Technology",  # 한미반도체 (semiconductor equipment)
    "047040": "Industrials",  # 대우건설 (construction)
    "055550": "Financials",  # 신한금융지주 (banking group)
    "062040": "Industrials",  # 산일전기 (transformers)
    "066570": "Consumer Discretionary",  # LG전자 (consumer electronics/appliances)
    "066970": "Materials",  # 엘앤에프 (battery cathode materials)
    "078930": "Industrials",  # GS (holding company)
    "086790": "Financials",  # 하나금융지주 (banking group)
    "093370": "Materials",  # 후성 (specialty chemicals)
    "096770": "Energy",  # SK이노베이션 (oil refining/petrochemicals)
    "103590": "Industrials",  # 일진전기 (electric cables/power equipment)
    "105560": "Financials",  # KB금융지주 (banking group)
    "128940": "Health Care",  # 한미약품 (pharmaceuticals)
    "181710": "Communication Services",  # NHN (games/internet services)
    "278470": "Consumer Discretionary",  # 에이피알 (beauty devices/cosmetics)
    "298040": "Industrials",  # 효성중공업 (heavy industrial equipment)
    "307950": "Information Technology",  # 현대오토에버 (automotive IT services)
    "317400": "Real Estate",  # 자이에스앤디 (building management/real estate services)
    "322000": "Industrials",  # 에이치디현대에너지솔루션 (solar panel manufacturing)
    "353200": "Information Technology",  # 대덕전자 (PCB)
    "402340": "Information Technology",  # SK스퀘어 (ICT/semiconductor investment holding)
    "475150": "Materials",  # 에스케이이터닉스 (battery recycling/materials)
}

if __name__ == "__main__":
    with engine.begin() as conn:
        for ticker, sector in SECTOR_OVERRIDES.items():
            conn.execute(
                text("UPDATE securities SET sector = :sector WHERE ticker = :ticker"),
                {"ticker": ticker, "sector": sector},
            )
    print(f"updated sector for {len(SECTOR_OVERRIDES)} tickers")
