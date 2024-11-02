# Global

`ours12`



# Local

`ours50` & `ours43`

avg9 = (50): $x$ 공간

avg2 = (43): $z$ 공간



# Global & Local

avg6,7,8 = (47,48,49) : $x$ 공간

- 47 = zero 실수
- 48 = zero 제대로 + bias 없음
- 49 = zero 제대로 + bias 있음



avg4,5 = (45,46): $z$ 공간

- 45 = zero 제대로 + bias 있음 v1 (local weight = torch.zeros)
- 56 = zero 제대로 + bias 있음 v2 (local weight = torch.ones)