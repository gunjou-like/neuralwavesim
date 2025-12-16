"""
API レスポンスの構造をテスト
"""

import requests
import json

def test_simulate_api():
    """API の /simulate エンドポイントをテスト"""
    
    print("=" * 60)
    print("🔍 API レスポンステスト")
    print("=" * 60)
    
    # リクエストペイロード
    payload = {
        "model_type": "physics",
        "physics_params": {
            "nx": 50,
            "nt": 100,
            "c": 1.0,
            "L": 10.0,
            "T": 10.0
        },
        "initial_condition": {
            "wave_type": "gaussian",
            "center": 5.0,
            "width": 1.0,
            "height": 1.0
        }
    }
    
    print("\n📤 リクエスト:")
    print(json.dumps(payload, indent=2))
    
    try:
        # API リクエスト
        response = requests.post(
            "http://localhost:8080/simulate",
            json=payload,
            timeout=30
        )
        
        print(f"\n✅ ステータスコード: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            print("\n📥 レスポンス構造:")
            print(f"  • success: {data.get('success')}")
            print(f"  • model_type: {data.get('model_type')}")
            print(f"  • computation_time: {data.get('computation_time'):.2f}ms")
            
            print("\n📊 results キー:")
            results = data.get("results", {})
            for key in results.keys():
                value = results[key]
                if isinstance(value, list):
                    if len(value) > 0 and isinstance(value[0], list):
                        print(f"  • {key}: 2D array, shape=({len(value)}, {len(value[0])})")
                    else:
                        print(f"  • {key}: 1D array, length={len(value)}")
                else:
                    print(f"  • {key}: {type(value).__name__} = {value}")
            
            print("\n📋 metadata:")
            metadata = data.get("metadata", {})
            for key, value in metadata.items():
                print(f"  • {key}: {value}")
            
            # wave_history の確認
            if "wave_history" in results:
                wave_history = results["wave_history"]
                print(f"\n✅ wave_history 存在確認:")
                print(f"  • Type: {type(wave_history)}")
                print(f"  • Shape: ({len(wave_history)}, {len(wave_history[0])})")
                print(f"  • Min: {min(min(row) for row in wave_history):.4f}")
                print(f"  • Max: {max(max(row) for row in wave_history):.4f}")
            else:
                print(f"\n❌ wave_history キーが存在しません")
                print(f"   利用可能なキー: {list(results.keys())}")
            
            return data
            
        else:
            print(f"\n❌ エラーレスポンス:")
            print(response.text)
            return None
            
    except requests.exceptions.ConnectionError:
        print("\n❌ API サーバーに接続できません")
        print("   docker-compose up でサーバーを起動してください")
        return None
    except Exception as e:
        print(f"\n❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🚀 API レスポンステスト開始\n")
    result = test_simulate_api()
    
    if result:
        print("\n" + "=" * 60)
        print("✅ テスト成功")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ テスト失敗")
        print("=" * 60)