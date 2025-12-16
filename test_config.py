"""
Configuration のテストスクリプト
InitialCondition の属性名を確認
"""

from core.config import PhysicsParams, InitialCondition, SimulationConfig
import json

def test_initial_condition():
    """InitialCondition の属性名をテスト"""
    print("=" * 60)
    print("🔍 InitialCondition 属性チェック")
    print("=" * 60)
    
    # InitialCondition を作成
    ic = InitialCondition(
        wave_type="gaussian",
        center=5.0,
        width=1.0,
        height=1.0
    )
    
    print(f"\n✅ InitialCondition 作成成功")
    print(f"   オブジェクト: {ic}")
    
    # すべての属性をリスト
    print(f"\n📋 利用可能な属性:")
    for attr in dir(ic):
        if not attr.startswith('_'):
            try:
                value = getattr(ic, attr)
                if not callable(value):
                    print(f"   • {attr}: {value}")
            except:
                pass
    
    # 'type' 属性の確認
    print(f"\n🔍 'type' 属性の確認:")
    if hasattr(ic, 'type'):
        print(f"   ✅ ic.type = {ic.type}")
    else:
        print(f"   ❌ 'type' 属性は存在しません")
    
    # 'wave_type' 属性の確認
    print(f"\n🔍 'wave_type' 属性の確認:")
    if hasattr(ic, 'wave_type'):
        print(f"   ✅ ic.wave_type = {ic.wave_type}")
    else:
        print(f"   ❌ 'wave_type' 属性は存在しません")
    
    return ic

def test_simulation_config():
    """SimulationConfig のテスト"""
    print("\n" + "=" * 60)
    print("🔍 SimulationConfig 属性チェック")
    print("=" * 60)
    
    config = SimulationConfig(
        model_type="physics",
        physics_params={"nx": 100, "nt": 200, "c": 1.0},
        initial_condition={"wave_type": "gaussian", "center": 5.0}
    )
    
    print(f"\n✅ SimulationConfig 作成成功")
    print(f"\n📋 Config の属性:")
    print(f"   • model_type: {config.model_type}")
    print(f"   • physics_params: {config.physics_params}")
    print(f"   • initial_condition: {config.initial_condition}")
    
    # InitialCondition の型を確認
    print(f"\n🔍 initial_condition の型:")
    print(f"   • type: {type(config.initial_condition)}")
    print(f"   • is InitialCondition: {isinstance(config.initial_condition, InitialCondition)}")
    
    # 属性アクセステスト
    print(f"\n🔍 属性アクセステスト:")
    try:
        wave_type = config.initial_condition.wave_type
        print(f"   ✅ config.initial_condition.wave_type = {wave_type}")
    except AttributeError as e:
        print(f"   ❌ エラー: {e}")
    
    try:
        ic_type = config.initial_condition.type
        print(f"   ✅ config.initial_condition.type = {ic_type}")
    except AttributeError as e:
        print(f"   ❌ エラー: {e}")
    
    return config

def test_json_serialization():
    """JSON シリアライズのテスト"""
    print("\n" + "=" * 60)
    print("🔍 JSON シリアライズテスト")
    print("=" * 60)
    
    ic = InitialCondition(wave_type="gaussian", center=5.0)
    
    # model_dump() でディクショナリに変換
    ic_dict = ic.model_dump()
    print(f"\n✅ model_dump() 結果:")
    print(json.dumps(ic_dict, indent=2))
    
    # JSON 文字列に変換
    ic_json = ic.model_dump_json()
    print(f"\n✅ model_dump_json() 結果:")
    print(ic_json)
    
    return ic_dict

if __name__ == "__main__":
    print("🚀 Configuration デバッグテスト開始\n")
    
    # テスト実行
    ic = test_initial_condition()
    config = test_simulation_config()
    ic_dict = test_json_serialization()
    
    print("\n" + "=" * 60)
    print("✅ すべてのテスト完了")
    print("=" * 60)