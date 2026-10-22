#!/usr/bin/env bash
t() { curl -s localhost:10042/translate -H "Content-Type: application/json" -d "$1" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d[\"source\"],\"->\",d[\"target\"],\"|\",d[\"seconds\"],\"s |\",d[\"translation\"])"; }
t "{\"text\":\"Your appointment at the hospital has been moved to Tuesday at 10am. Please bring your medication list.\",\"target\":\"cy\"}"
t "{\"text\":\"The council will collect recycling every other Wednesday from next month.\",\"target\":\"cy\"}"
t "{\"text\":\"If you need help filling in this form, please call us on 0300 123 4567.\",\"target\":\"cy\"}"
t "{\"text\":\"Children under five travel free on all buses in Wales.\",\"target\":\"cy\"}"
